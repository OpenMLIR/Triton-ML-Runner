#define PY_SSIZE_T_CLEAN

#include <Python.h>

#define TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API 1

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/string.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace triton_runner_tvm_ffi {

using tvm::ffi::Any;
using tvm::ffi::AnyView;
using tvm::ffi::Array;
using tvm::ffi::Bytes;
using tvm::ffi::Function;
using tvm::ffi::ObjectRef;
using tvm::ffi::PackedArgs;
using tvm::ffi::String;
using tvm::ffi::TensorView;

namespace {

constexpr const char* kDLPackExchangeApiCapsuleName = "dlpack_exchange_api";
constexpr const char* kDLPackCapsuleName = "dltensor";
constexpr const char* kUsedDLPackCapsuleName = "used_dltensor";
constexpr const char* kDLPackVersionedCapsuleName = "dltensor_versioned";
constexpr const char* kUsedDLPackVersionedCapsuleName = "used_dltensor_versioned";

class PyObjectRef {
 public:
  explicit PyObjectRef(PyObject* obj = nullptr) : obj_(obj) {}
  PyObjectRef(const PyObjectRef&) = delete;
  PyObjectRef& operator=(const PyObjectRef&) = delete;
  PyObjectRef(PyObjectRef&& other) noexcept : obj_(other.obj_) { other.obj_ = nullptr; }
  PyObjectRef& operator=(PyObjectRef&& other) noexcept {
    if (this != &other) {
      if (obj_ != nullptr) {
        Py_DECREF(obj_);
      }
      obj_ = other.obj_;
      other.obj_ = nullptr;
    }
    return *this;
  }
  ~PyObjectRef() {
    if (obj_ != nullptr) {
      Py_DECREF(obj_);
    }
  }

  PyObject* get() const { return obj_; }
  explicit operator bool() const { return obj_ != nullptr; }

 private:
  PyObject* obj_ = nullptr;
};

class ScopedPyGIL {
 public:
  ScopedPyGIL() : state_(PyGILState_Ensure()) {}
  ~ScopedPyGIL() { PyGILState_Release(state_); }

 private:
  PyGILState_STATE state_;
};

std::string ConsumePythonErrorString() {
  if (!PyErr_Occurred()) {
    return std::string();
  }
  PyObject *exc_type = nullptr, *exc_value = nullptr, *exc_tb = nullptr;
  PyErr_Fetch(&exc_type, &exc_value, &exc_tb);
  PyErr_NormalizeException(&exc_type, &exc_value, &exc_tb);
  PyObjectRef type_ref(exc_type);
  PyObjectRef value_ref(exc_value);
  PyObjectRef tb_ref(exc_tb);
  if (!value_ref) {
    return "unknown Python error";
  }
  PyObjectRef text(PyObject_Str(value_ref.get()));
  if (!text) {
    PyErr_Clear();
    return "unknown Python error";
  }
  const char* utf8 = PyUnicode_AsUTF8(text.get());
  if (utf8 == nullptr) {
    PyErr_Clear();
    return "unknown Python error";
  }
  return std::string(utf8);
}

[[noreturn]] void ThrowPythonError(const char* context) {
  std::string detail = ConsumePythonErrorString();
  if (detail.empty()) {
    TVM_FFI_THROW(RuntimeError) << context;
  }
  TVM_FFI_THROW(RuntimeError) << context << ": " << detail;
}

PyObject* GetRequiredAttr(PyObject* obj, const char* attr_name, const char* context) {
  PyObject* attr = PyObject_GetAttrString(obj, attr_name);
  if (attr == nullptr) {
    ThrowPythonError(context);
  }
  return attr;
}

int64_t PyLongToInt64(PyObject* obj, const char* context) {
  long long value = PyLong_AsLongLong(obj);
  if (value == -1 && PyErr_Occurred()) {
    ThrowPythonError(context);
  }
  return static_cast<int64_t>(value);
}

bool PaddingIsNan(PyObject* padding_obj) {
  if (padding_obj == Py_None) {
    return false;
  }
  if (!PyUnicode_Check(padding_obj)) {
    return false;
  }
  int cmp = PyUnicode_CompareWithASCIIString(padding_obj, "nan");
  if (cmp == -1 && PyErr_Occurred()) {
    ThrowPythonError("Failed to inspect tensor descriptor padding");
  }
  return cmp == 0;
}

int ParseTensorDescRank(std::string_view type_name) {
  size_t open = type_name.find('[');
  size_t close = type_name.rfind(']');
  TVM_FFI_CHECK(open != std::string_view::npos && close != std::string_view::npos && close > open, TypeError)
      << "Unsupported tensordesc Triton signature: " << type_name;
  std::string_view shape_sig = type_name.substr(open + 1, close - open - 1);
  if (shape_sig.empty()) {
    return 1;
  }
  int rank = 1;
  for (char ch : shape_sig) {
    if (ch == ',') {
      ++rank;
    }
  }
  return rank;
}

void DecrefOpaquePyObject(void* handle) {
  if (handle == nullptr) {
    return;
  }
  PyGILState_STATE gil_state = PyGILState_Ensure();
  Py_DECREF(reinterpret_cast<PyObject*>(handle));
  PyGILState_Release(gil_state);
}

ObjectRef ObjectRefFromOwnedHandle(TVMFFIObjectHandle handle) {
  auto ptr = tvm::ffi::details::ObjectUnsafe::ObjectPtrFromOwned<tvm::ffi::Object>(
      reinterpret_cast<TVMFFIObject*>(handle));
  return tvm::ffi::details::ObjectUnsafe::ObjectRefFromObjectPtr<ObjectRef>(std::move(ptr));
}

ObjectRef MakeOpaquePyObjectRef(PyObject* py_obj) {
  TVM_FFI_CHECK(py_obj != nullptr, ValueError) << "Opaque Python object must not be null.";
  TVMFFIObjectHandle handle = nullptr;
  Py_INCREF(py_obj);
  if (TVMFFIObjectCreateOpaque(py_obj, kTVMFFIOpaquePyObject, DecrefOpaquePyObject, &handle) != 0) {
    Py_DECREF(py_obj);
    ThrowPythonError("Failed to create ffi.OpaquePyObject");
  }
  return ObjectRefFromOwnedHandle(handle);
}

PyObject* OpaquePyObjectBorrowedFromAny(AnyView arg, int64_t arg_index) {
  ObjectRef obj = arg.cast<ObjectRef>();
  TVM_FFI_CHECK(obj.type_index() == tvm::ffi::TypeIndex::kTVMFFIOpaquePyObject, TypeError)
      << "Tensor descriptor argument #" << arg_index
      << " must arrive as ffi.OpaquePyObject, got type_index=" << obj.type_index();
  TVMFFIObject* handle = tvm::ffi::details::ObjectUnsafe::TVMFFIObjectPtrFromObjectRef(obj);
  auto* cell = TVMFFIOpaqueObjectGetCellPtr(handle);
  TVM_FFI_CHECK(cell != nullptr && cell->handle != nullptr, TypeError)
      << "Tensor descriptor argument #" << arg_index << " does not hold a Python object.";
  return reinterpret_cast<PyObject*>(cell->handle);
}

struct StreamContextInfo {
  bool present = false;
  int32_t device_type = -1;
  int32_t device_id = -1;
  TVMFFIStreamHandle stream = nullptr;
};

class ScopedEnvStream {
 public:
  explicit ScopedEnvStream(const StreamContextInfo& info) : info_(info) {
    if (!info_.present) {
      return;
    }
    if (TVMFFIEnvSetStream(info_.device_type, info_.device_id, info_.stream, &previous_stream_) != 0) {
      ThrowPythonError("Failed to set TVM-FFI stream context");
    }
    active_ = true;
  }

  ~ScopedEnvStream() {
    if (!active_) {
      return;
    }
    if (previous_stream_ != info_.stream) {
      (void)TVMFFIEnvSetStream(info_.device_type, info_.device_id, previous_stream_, nullptr);
    }
  }

 private:
  StreamContextInfo info_;
  TVMFFIStreamHandle previous_stream_ = nullptr;
  bool active_ = false;
};

void MaybeCaptureCurrentStream(const DLPackExchangeAPI* exchange_api,
                               const DLDevice& device,
                               StreamContextInfo* stream_ctx) {
  if (stream_ctx == nullptr || exchange_api == nullptr || exchange_api->current_work_stream == nullptr ||
      device.device_type == kDLCPU) {
    return;
  }
  if (stream_ctx->present) {
    TVM_FFI_CHECK(stream_ctx->device_type == device.device_type && stream_ctx->device_id == device.device_id,
                  ValueError)
        << "All tensor descriptor bases must live on the same device.";
    return;
  }
  void* stream = nullptr;
  if (exchange_api->current_work_stream(device.device_type, device.device_id, &stream) != 0) {
    ThrowPythonError("Failed to query current producer stream from DLPack exchange API");
  }
  stream_ctx->present = true;
  stream_ctx->device_type = device.device_type;
  stream_ctx->device_id = device.device_id;
  stream_ctx->stream = stream;
}

ObjectRef TensorObjectRefFromDLPackExchangeApi(PyObject* tensor_obj, StreamContextInfo* stream_ctx) {
  PyObjectRef capsule(
      PyObject_GetAttrString(reinterpret_cast<PyObject*>(Py_TYPE(tensor_obj)), "__dlpack_c_exchange_api__"));
  if (!capsule) {
    ThrowPythonError("Failed to load __dlpack_c_exchange_api__ from tensor type");
  }
  const auto* exchange_api = reinterpret_cast<const DLPackExchangeAPI*>(
      PyCapsule_GetPointer(capsule.get(), kDLPackExchangeApiCapsuleName));
  if (exchange_api == nullptr) {
    ThrowPythonError("Failed to access dlpack_exchange_api capsule");
  }

  DLManagedTensorVersioned* managed_tensor = nullptr;
  if (exchange_api->managed_tensor_from_py_object_no_sync == nullptr ||
      exchange_api->managed_tensor_from_py_object_no_sync(tensor_obj, &managed_tensor) != 0) {
    ThrowPythonError("Failed to convert tensor descriptor base via DLPack exchange API");
  }
  MaybeCaptureCurrentStream(exchange_api, managed_tensor->dl_tensor.device, stream_ctx);

  TVMFFIObjectHandle handle = nullptr;
  if (TVMFFITensorFromDLPackVersioned(managed_tensor, 0, 0, &handle) != 0) {
    if (managed_tensor->deleter != nullptr) {
      managed_tensor->deleter(managed_tensor);
    }
    ThrowPythonError("Failed to convert DLManagedTensorVersioned into ffi.Tensor");
  }
  return ObjectRefFromOwnedHandle(handle);
}

ObjectRef TensorObjectRefFromLegacyDLPack(PyObject* tensor_obj) {
  PyObjectRef capsule(PyObject_CallMethod(tensor_obj, "__dlpack__", nullptr));
  if (!capsule) {
    ThrowPythonError("Failed to call tensor.__dlpack__()");
  }

  TVMFFIObjectHandle handle = nullptr;
  if (PyCapsule_IsValid(capsule.get(), kDLPackVersionedCapsuleName)) {
    auto* managed = reinterpret_cast<DLManagedTensorVersioned*>(
        PyCapsule_GetPointer(capsule.get(), kDLPackVersionedCapsuleName));
    if (managed == nullptr) {
      ThrowPythonError("Failed to access dltensor_versioned capsule");
    }
    if (TVMFFITensorFromDLPackVersioned(managed, 0, 0, &handle) != 0) {
      ThrowPythonError("Failed to convert dltensor_versioned capsule into ffi.Tensor");
    }
    if (PyCapsule_SetDestructor(capsule.get(), nullptr) != 0 ||
        PyCapsule_SetName(capsule.get(), kUsedDLPackVersionedCapsuleName) != 0) {
      ThrowPythonError("Failed to mark dltensor_versioned capsule as consumed");
    }
    return ObjectRefFromOwnedHandle(handle);
  }

  TVM_FFI_CHECK(PyCapsule_IsValid(capsule.get(), kDLPackCapsuleName), TypeError)
      << "tensor.__dlpack__() must return a dltensor or dltensor_versioned capsule.";
  auto* managed = reinterpret_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.get(), kDLPackCapsuleName));
  if (managed == nullptr) {
    ThrowPythonError("Failed to access dltensor capsule");
  }
  if (TVMFFITensorFromDLPack(managed, 0, 0, &handle) != 0) {
    ThrowPythonError("Failed to convert dltensor capsule into ffi.Tensor");
  }
  if (PyCapsule_SetDestructor(capsule.get(), nullptr) != 0 ||
      PyCapsule_SetName(capsule.get(), kUsedDLPackCapsuleName) != 0) {
    ThrowPythonError("Failed to mark dltensor capsule as consumed");
  }
  return ObjectRefFromOwnedHandle(handle);
}

ObjectRef TensorBaseToObjectRef(PyObject* base_obj, StreamContextInfo* stream_ctx) {
  int has_exchange_api =
      PyObject_HasAttrString(reinterpret_cast<PyObject*>(Py_TYPE(base_obj)), "__dlpack_c_exchange_api__");
  if (has_exchange_api == -1) {
    ThrowPythonError("Failed to inspect tensor type for __dlpack_c_exchange_api__");
  }
  if (has_exchange_api == 1) {
    return TensorObjectRefFromDLPackExchangeApi(base_obj, stream_ctx);
  }

  int has_dlpack = PyObject_HasAttrString(base_obj, "__dlpack__");
  if (has_dlpack == -1) {
    ThrowPythonError("Failed to inspect tensor descriptor base for __dlpack__");
  }
  if (has_dlpack == 1) {
    return TensorObjectRefFromLegacyDLPack(base_obj);
  }
  return MakeOpaquePyObjectRef(base_obj);
}

void AppendTensorDescExpandedArgs(PyObject* tensor_desc,
                                  int rank,
                                  int64_t arg_index,
                                  std::vector<Any>* owned_args,
                                  std::vector<AnyView>* launch_args,
                                  StreamContextInfo* stream_ctx) {
  PyObjectRef base(
      GetRequiredAttr(tensor_desc, "base", "Tensor descriptor argument is missing required attribute 'base'"));
  PyObjectRef shape(
      GetRequiredAttr(tensor_desc, "shape", "Tensor descriptor argument is missing required attribute 'shape'"));
  PyObjectRef strides(GetRequiredAttr(
      tensor_desc, "strides", "Tensor descriptor argument is missing required attribute 'strides'"));
  PyObject* padding_obj = PyObject_GetAttrString(tensor_desc, "padding");
  if (padding_obj == nullptr && PyErr_ExceptionMatches(PyExc_AttributeError)) {
    PyErr_Clear();
    padding_obj = Py_None;
    Py_INCREF(padding_obj);
  } else if (padding_obj == nullptr) {
    ThrowPythonError("Failed to read tensor descriptor padding");
  }
  PyObjectRef padding(padding_obj);

  PyObjectRef shape_fast(PySequence_Fast(shape.get(), "tensor descriptor shape must be a sequence"));
  if (!shape_fast) {
    ThrowPythonError("Failed to iterate tensor descriptor shape");
  }
  PyObjectRef strides_fast(PySequence_Fast(strides.get(), "tensor descriptor strides must be a sequence"));
  if (!strides_fast) {
    ThrowPythonError("Failed to iterate tensor descriptor strides");
  }
  TVM_FFI_CHECK(PySequence_Fast_GET_SIZE(shape_fast.get()) == rank, ValueError)
      << "Tensor descriptor argument #" << arg_index << " expected rank " << rank << ", got shape of length "
      << PySequence_Fast_GET_SIZE(shape_fast.get()) << ".";
  TVM_FFI_CHECK(PySequence_Fast_GET_SIZE(strides_fast.get()) == rank, ValueError)
      << "Tensor descriptor argument #" << arg_index << " expected rank " << rank
      << ", got strides of length " << PySequence_Fast_GET_SIZE(strides_fast.get()) << ".";

  owned_args->emplace_back(TensorBaseToObjectRef(base.get(), stream_ctx));
  launch_args->push_back(owned_args->back());
  for (int i = 0; i < rank; ++i) {
    owned_args->emplace_back(
        PyLongToInt64(PySequence_Fast_GET_ITEM(shape_fast.get(), i), "Tensor descriptor shape entries must be ints"));
    launch_args->push_back(owned_args->back());
  }
  for (int i = 0; i < rank; ++i) {
    owned_args->emplace_back(PyLongToInt64(
        PySequence_Fast_GET_ITEM(strides_fast.get(), i), "Tensor descriptor stride entries must be ints"));
    launch_args->push_back(owned_args->back());
  }
  owned_args->emplace_back(PaddingIsNan(padding.get()));
  launch_args->push_back(owned_args->back());
}

}  // namespace

inline void CheckCudaRuntimeError(cudaError_t err) {
  if (err != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "CUDA Runtime Error: " << cudaGetErrorName(err) << " (" << static_cast<int>(err)
        << "): " << cudaGetErrorString(err);
  }
}

inline void CheckCudaDriverError(CUresult err) {
  if (err != CUDA_SUCCESS) {
    const char* err_name = nullptr;
    const char* err_string = nullptr;
    cuGetErrorName(err, &err_name);
    cuGetErrorString(err, &err_string);
    TVM_FFI_THROW(RuntimeError)
        << "CUDA Driver Error: " << (err_name != nullptr ? err_name : "<unknown>") << " ("
        << static_cast<int>(err) << "): " << (err_string != nullptr ? err_string : "<unknown>");
  }
}

#define TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(stmt) \
  do {                                                       \
    ::cudaError_t __err = (stmt);                            \
    ::triton_runner_tvm_ffi::CheckCudaRuntimeError(__err);   \
  } while (0)

#define TVM_FFI_CHECK_TRITON_RUNNER_CUDA_DRIVER_ERROR(stmt) \
  do {                                                      \
    ::CUresult __err = (stmt);                              \
    ::triton_runner_tvm_ffi::CheckCudaDriverError(__err);   \
  } while (0)

using cuTensorMapEncodeTiled_t = CUresult (*)(
    CUtensorMap*,
    CUtensorMapDataType,
    cuuint32_t,
    void*,
    const cuuint64_t*,
    const cuuint64_t*,
    const cuuint32_t*,
    const cuuint32_t*,
    CUtensorMapInterleave,
    CUtensorMapSwizzle,
    CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill);

inline cuTensorMapEncodeTiled_t GetCuTensorMapEncodeTiledHandle() {
  static void* lib_handle = nullptr;
  static cuTensorMapEncodeTiled_t func = nullptr;
  if (func != nullptr) {
    return func;
  }
  if (lib_handle == nullptr) {
    lib_handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    TVM_FFI_CHECK(lib_handle != nullptr, RuntimeError)
        << "Failed to open libcuda.so.1 for tensor descriptor support.";
  }
  dlerror();
  func = reinterpret_cast<cuTensorMapEncodeTiled_t>(dlsym(lib_handle, "cuTensorMapEncodeTiled"));
  const char* err = dlerror();
  TVM_FFI_CHECK(err == nullptr && func != nullptr, RuntimeError)
      << "Failed to retrieve cuTensorMapEncodeTiled from libcuda.so.1.";
  return func;
}

inline void FillTmaDescriptor(CUtensorMap* tensor_map,
                              void* global_address,
                              int swizzle,
                              int elem_size,
                              int elem_type,
                              const uint32_t* block_size,
                              int rank,
                              const int64_t* shape,
                              const int64_t* strides,
                              bool padding_nan,
                              bool fp4_padded,
                              const char* arg_name) {
  TVM_FFI_CHECK(rank > 0 && rank <= 5, ValueError)
      << "Tensor descriptor " << arg_name << " has unsupported rank " << rank;
  TVM_FFI_CHECK(strides[rank - 1] == 1, ValueError)
      << "Tensor descriptor " << arg_name << " requires innermost stride == 1.";

  uint32_t block_size_int[5] = {1, 1, 1, 1, 1};
  uint64_t shape_int[5] = {1, 1, 1, 1, 1};
  uint64_t strides_bytes[5] = {0, 0, 0, 0, 0};
  uint32_t element_strides[5] = {1, 1, 1, 1, 1};

  for (int i = 0; i < rank; ++i) {
    TVM_FFI_CHECK(shape[i] >= 0, ValueError)
        << "Tensor descriptor " << arg_name << " shape[" << i << "] must be non-negative.";
    int reversed = rank - i - 1;
    uint64_t dim = static_cast<uint64_t>(shape[i]);
    if (fp4_padded && i == rank - 1) {
      dim *= 2;
    }
    shape_int[reversed] = dim;
    block_size_int[reversed] = block_size[i];
  }

  for (int i = 0; i + 1 < rank; ++i) {
    TVM_FFI_CHECK(strides[i] >= 0, ValueError)
        << "Tensor descriptor " << arg_name << " stride[" << i << "] must be non-negative.";
    int reversed = rank - i - 2;
    strides_bytes[reversed] = static_cast<uint64_t>(elem_size) * static_cast<uint64_t>(strides[i]);
  }
  strides_bytes[rank - 1] =
      shape_int[rank - 1] * static_cast<uint64_t>(rank == 1 ? elem_size : strides_bytes[rank - 2]);

  CUtensorMapFloatOOBfill fill =
      padding_nan ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
  auto encode = GetCuTensorMapEncodeTiledHandle();
  auto result = encode(
      tensor_map,
      static_cast<CUtensorMapDataType>(elem_type),
      static_cast<cuuint32_t>(rank),
      global_address,
      shape_int,
      strides_bytes,
      block_size_int,
      element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      static_cast<CUtensorMapSwizzle>(swizzle),
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      fill);
  TVM_FFI_CHECK_TRITON_RUNNER_CUDA_DRIVER_ERROR(result);
}

struct ScratchBuffer {
  void* base = nullptr;
  void* aligned = nullptr;
  size_t capacity = 0;
  size_t alignment = 1;
};

static std::mutex g_scratch_mu;
static std::unordered_map<int, ScratchBuffer> g_global_scratch_buffers;
static std::unordered_map<int, ScratchBuffer> g_profile_scratch_buffers;

inline void* AlignPtr(void* ptr, size_t alignment) {
  uintptr_t raw = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t mask = alignment - 1;
  uintptr_t aligned = (raw + mask) & ~mask;
  return reinterpret_cast<void*>(aligned);
}

inline void* GetScratchBuffer(std::unordered_map<int, ScratchBuffer>& buffers,
                              int device_id,
                              size_t size,
                              size_t alignment) {
  if (size == 0) {
    return nullptr;
  }
  if (alignment == 0) {
    alignment = 1;
  }
  std::lock_guard<std::mutex> guard(g_scratch_mu);
  auto& buffer = buffers[device_id];
  if (buffer.base == nullptr || buffer.capacity < size || buffer.alignment < alignment) {
    int previous_device = -1;
    TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaGetDevice(&previous_device));
    if (previous_device != device_id) {
      TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaSetDevice(device_id));
    }
    if (buffer.base != nullptr) {
      TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaFree(buffer.base));
    }
    void* base = nullptr;
    TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaMalloc(&base, size + alignment - 1));
    buffer.base = base;
    buffer.aligned = AlignPtr(base, alignment);
    buffer.capacity = size;
    buffer.alignment = alignment;
    if (previous_device != device_id) {
      TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaSetDevice(previous_device));
    }
  }
  return buffer.aligned;
}

enum class ArgKind {
  kPointer,
  kI1,
  kI8,
  kI16,
  kI32,
  kI64,
  kU1,
  kU8,
  kU16,
  kU32,
  kU64,
  kFp16,
  kBf16,
  kFp32,
  kFp64,
  kTensorDesc,
  kNvTmaDesc,
};

struct TensorDescMetadata {
  bool present = false;
  int swizzle = 0;
  int elem_size = 0;
  int elem_type = 0;
  bool fp4_padded = false;
  std::array<uint32_t, 5> block_size = {1, 1, 1, 1, 1};
};

struct RuntimeArgSpec {
  std::string name;
  ArgKind kind;
  int rank = 0;
  TensorDescMetadata tensordesc_meta;
};

struct RegisteredKernel {
  CUfunction function = nullptr;
  uint32_t block_x = 0;
  uint32_t shared_memory = 0;
  size_t global_scratch_size = 0;
  size_t global_scratch_align = 1;
  size_t profile_scratch_size = 0;
  size_t profile_scratch_align = 1;
  std::vector<RuntimeArgSpec> runtime_args;
};

struct BoundArgsLauncherPlan {
  Function tvm_func;
  int64_t registry_handle = 0;
  std::vector<int32_t> fast_arg_indices;
  int32_t total_bound_arg_count = 0;
};

static std::mutex g_kernel_mu;
static std::unordered_map<int64_t, std::unique_ptr<RegisteredKernel>> g_registered_kernels;

struct alignas(128) KernelArgSlot {
  union {
    void* ptr;
    bool b;
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    uint8_t u8;
    uint16_t u16;
    uint32_t u32;
    uint64_t u64;
    __half fp16;
    __nv_bfloat16 bf16;
    float fp32;
    double fp64;
    CUtensorMap tensor_map;
  };
};

inline ArgKind ArgKindFromCode(int64_t code) {
  switch (code) {
    case 0:
      return ArgKind::kPointer;
    case 1:
      return ArgKind::kI1;
    case 2:
      return ArgKind::kI8;
    case 3:
      return ArgKind::kI16;
    case 4:
      return ArgKind::kI32;
    case 5:
      return ArgKind::kI64;
    case 6:
      return ArgKind::kU1;
    case 7:
      return ArgKind::kU8;
    case 8:
      return ArgKind::kU16;
    case 9:
      return ArgKind::kU32;
    case 10:
      return ArgKind::kU64;
    case 11:
      return ArgKind::kFp16;
    case 12:
      return ArgKind::kBf16;
    case 13:
      return ArgKind::kFp32;
    case 14:
      return ArgKind::kFp64;
    case 15:
      return ArgKind::kTensorDesc;
    case 16:
      return ArgKind::kNvTmaDesc;
    default:
      TVM_FFI_THROW(ValueError) << "Unsupported Triton runtime arg kind code: " << code;
      return ArgKind::kPointer;
  }
}

inline RegisteredKernel* GetRegisteredKernel(int64_t handle) {
  auto* kernel = reinterpret_cast<RegisteredKernel*>(static_cast<uintptr_t>(handle));
  TVM_FFI_CHECK(kernel != nullptr, KeyError)
      << "No registered TVM-FFI Triton kernel for handle=" << handle;
  return kernel;
}

inline void FillRegisteredKernelRuntimeArgs(
    RegisteredKernel* kernel,
    int64_t runtime_arg_count,
    const Array<String>& runtime_arg_names,
    const Array<int64_t>& runtime_arg_kind_codes,
    const Array<int64_t>& runtime_arg_ranks,
    const Array<int64_t>& runtime_arg_meta_present,
    const Array<int64_t>& runtime_arg_swizzles,
    const Array<int64_t>& runtime_arg_elem_sizes,
    const Array<int64_t>& runtime_arg_elem_types,
    const Array<int64_t>& runtime_arg_fp4_padded,
    const Array<int64_t>& runtime_arg_block_size_offsets,
    const Array<int64_t>& runtime_arg_block_size_values) {
  TVM_FFI_CHECK(runtime_arg_kind_codes.size() == runtime_arg_count, ValueError)
      << "runtime_arg_kind_codes size mismatch";
  TVM_FFI_CHECK(runtime_arg_ranks.size() == runtime_arg_count, ValueError)
      << "runtime_arg_ranks size mismatch";
  TVM_FFI_CHECK(runtime_arg_meta_present.size() == runtime_arg_count, ValueError)
      << "runtime_arg_meta_present size mismatch";
  TVM_FFI_CHECK(runtime_arg_swizzles.size() == runtime_arg_count, ValueError)
      << "runtime_arg_swizzles size mismatch";
  TVM_FFI_CHECK(runtime_arg_elem_sizes.size() == runtime_arg_count, ValueError)
      << "runtime_arg_elem_sizes size mismatch";
  TVM_FFI_CHECK(runtime_arg_elem_types.size() == runtime_arg_count, ValueError)
      << "runtime_arg_elem_types size mismatch";
  TVM_FFI_CHECK(runtime_arg_fp4_padded.size() == runtime_arg_count, ValueError)
      << "runtime_arg_fp4_padded size mismatch";
  TVM_FFI_CHECK(runtime_arg_block_size_offsets.size() == runtime_arg_count + 1, ValueError)
      << "runtime_arg_block_size_offsets size mismatch";

  kernel->runtime_args.reserve(static_cast<size_t>(runtime_arg_count));
  for (int64_t i = 0; i < runtime_arg_count; ++i) {
    RuntimeArgSpec spec;
    spec.name = runtime_arg_names[i];
    spec.kind = ArgKindFromCode(runtime_arg_kind_codes[i]);
    spec.rank = static_cast<int>(runtime_arg_ranks[i]);
    int64_t block_offset = runtime_arg_block_size_offsets[i];
    int64_t block_end = runtime_arg_block_size_offsets[i + 1];
    TVM_FFI_CHECK(block_offset >= 0 && block_end >= block_offset, ValueError)
        << "Invalid tensor descriptor block_size offsets for " << spec.name;
    TVM_FFI_CHECK(block_end <= runtime_arg_block_size_values.size(), ValueError)
        << "Tensor descriptor block_size offset out of bounds for " << spec.name;

    if (spec.kind == ArgKind::kTensorDesc) {
      TVM_FFI_CHECK(spec.rank > 0 && spec.rank <= 5, ValueError)
          << "Tensor descriptor " << spec.name << " has unsupported rank " << spec.rank;
      if (runtime_arg_meta_present[i] != 0) {
        TVM_FFI_CHECK(block_end - block_offset == spec.rank, ValueError)
            << "Tensor descriptor block_size rank mismatch for " << spec.name;
        spec.tensordesc_meta.present = true;
        spec.tensordesc_meta.swizzle = static_cast<int>(runtime_arg_swizzles[i]);
        spec.tensordesc_meta.elem_size = static_cast<int>(runtime_arg_elem_sizes[i]);
        spec.tensordesc_meta.elem_type = static_cast<int>(runtime_arg_elem_types[i]);
        spec.tensordesc_meta.fp4_padded = runtime_arg_fp4_padded[i] != 0;
        for (int j = 0; j < spec.rank; ++j) {
          spec.tensordesc_meta.block_size[static_cast<size_t>(j)] =
              static_cast<uint32_t>(runtime_arg_block_size_values[block_offset + j]);
        }
      } else {
        TVM_FFI_CHECK(block_end == block_offset, ValueError)
            << "Tensor descriptor without metadata must not carry block_size values for " << spec.name;
      }
    } else {
      TVM_FFI_CHECK(spec.rank == 0, ValueError)
          << "Non-tensor runtime arg " << spec.name << " must have rank 0";
      TVM_FFI_CHECK(runtime_arg_meta_present[i] == 0, ValueError)
          << "Non-tensor runtime arg " << spec.name << " must not carry tensor metadata";
      TVM_FFI_CHECK(block_end == block_offset, ValueError)
          << "Non-tensor runtime arg " << spec.name << " must not carry block_size values";
    }
    kernel->runtime_args.push_back(std::move(spec));
  }
}

int64_t RegisterKernelFromFunction(int64_t function_handle,
                                   int64_t block_x,
                                   int64_t shared_memory,
                                   int64_t global_scratch_size,
                                   int64_t global_scratch_align,
                                   int64_t profile_scratch_size,
                                   int64_t profile_scratch_align,
                                   const Array<String>& runtime_arg_names,
                                   const Array<int64_t>& runtime_arg_kind_codes,
                                   const Array<int64_t>& runtime_arg_ranks,
                                   const Array<int64_t>& runtime_arg_meta_present,
                                   const Array<int64_t>& runtime_arg_swizzles,
                                   const Array<int64_t>& runtime_arg_elem_sizes,
                                   const Array<int64_t>& runtime_arg_elem_types,
                                   const Array<int64_t>& runtime_arg_fp4_padded,
                                   const Array<int64_t>& runtime_arg_block_size_offsets,
                                   const Array<int64_t>& runtime_arg_block_size_values) {
  auto kernel = std::make_unique<RegisteredKernel>();
  kernel->function = reinterpret_cast<CUfunction>(static_cast<uintptr_t>(function_handle));
  kernel->block_x = static_cast<uint32_t>(block_x);
  kernel->shared_memory = static_cast<uint32_t>(shared_memory);
  kernel->global_scratch_size = static_cast<size_t>(global_scratch_size);
  kernel->global_scratch_align = static_cast<size_t>(global_scratch_align);
  kernel->profile_scratch_size = static_cast<size_t>(profile_scratch_size);
  kernel->profile_scratch_align = static_cast<size_t>(profile_scratch_align);
  FillRegisteredKernelRuntimeArgs(
      kernel.get(),
      runtime_arg_names.size(),
      runtime_arg_names,
      runtime_arg_kind_codes,
      runtime_arg_ranks,
      runtime_arg_meta_present,
      runtime_arg_swizzles,
      runtime_arg_elem_sizes,
      runtime_arg_elem_types,
      runtime_arg_fp4_padded,
      runtime_arg_block_size_offsets,
      runtime_arg_block_size_values);

  int64_t handle = static_cast<int64_t>(reinterpret_cast<uintptr_t>(kernel.get()));
  {
    std::lock_guard<std::mutex> guard(g_kernel_mu);
    g_registered_kernels[handle] = std::move(kernel);
  }
  return handle;
}

template <typename ArgAccess>
inline void LaunchPackedImpl(int64_t registry_handle,
                             int32_t grid_x,
                             int32_t grid_y,
                             int32_t grid_z,
                             ArgAccess&& get_arg,
                             int32_t num_args,
                             Any* ret);

Function MakeBoundArgsLauncher(const Function& tvm_func,
                               int64_t registry_handle,
                               const Array<String>& signature_type_names) {
  TVM_FFI_CHECK(tvm_func != nullptr, ValueError) << "tvm_func must not be null.";

  auto plan = std::make_shared<BoundArgsLauncherPlan>();
  plan->tvm_func = tvm_func;
  plan->registry_handle = registry_handle;
  plan->total_bound_arg_count = static_cast<int32_t>(signature_type_names.size());

  int32_t arg_offset = 0;
  for (const String& type_name : signature_type_names) {
    std::string_view type_name_view(type_name.data(), type_name.size());
    if (type_name_view == "constexpr") {
      ++arg_offset;
      continue;
    }
    plan->fast_arg_indices.push_back(arg_offset);
    ++arg_offset;
  }

  return Function::FromPacked([plan = std::move(plan)](PackedArgs args, Any* ret) {
    TVM_FFI_CHECK(args.size() >= 3, ValueError)
        << "Expected at least 3 launch arguments, got " << args.size();
    TVM_FFI_CHECK(args.size() == plan->total_bound_arg_count + 3, ValueError)
        << "Expected " << plan->total_bound_arg_count << " bound arguments for TVM-FFI launch, got "
        << (args.size() - 3) << ".";

    auto get_arg = [&](int64_t idx) {
      return args[plan->fast_arg_indices[static_cast<size_t>(idx)] + 3];
    };
    LaunchPackedImpl(plan->registry_handle,
                     args[0].cast<int32_t>(),
                     args[1].cast<int32_t>(),
                     args[2].cast<int32_t>(),
                     get_arg,
                     static_cast<int32_t>(plan->fast_arg_indices.size()),
                     ret);
  });
}

inline void StoreIntegerArg(ArgKind kind, const Any& arg, std::vector<KernelArgSlot>* slots, std::vector<void*>* out) {
  slots->emplace_back();
  KernelArgSlot& slot = slots->back();
  switch (kind) {
    case ArgKind::kI1:
      slot.b = arg.cast<bool>();
      out->push_back(&slot.b);
      break;
    case ArgKind::kI8:
      slot.i8 = arg.cast<int8_t>();
      out->push_back(&slot.i8);
      break;
    case ArgKind::kI16:
      slot.i16 = arg.cast<int16_t>();
      out->push_back(&slot.i16);
      break;
    case ArgKind::kI32:
      slot.i32 = arg.cast<int32_t>();
      out->push_back(&slot.i32);
      break;
    case ArgKind::kI64:
      slot.i64 = arg.cast<int64_t>();
      out->push_back(&slot.i64);
      break;
    case ArgKind::kU1:
      slot.b = arg.cast<bool>();
      out->push_back(&slot.b);
      break;
    case ArgKind::kU8:
      slot.u8 = arg.cast<uint8_t>();
      out->push_back(&slot.u8);
      break;
    case ArgKind::kU16:
      slot.u16 = arg.cast<uint16_t>();
      out->push_back(&slot.u16);
      break;
    case ArgKind::kU32:
      slot.u32 = arg.cast<uint32_t>();
      out->push_back(&slot.u32);
      break;
    case ArgKind::kU64:
      slot.u64 = arg.cast<uint64_t>();
      out->push_back(&slot.u64);
      break;
    default:
      TVM_FFI_THROW(RuntimeError) << "Invalid integer arg kind";
  }
}

inline void StoreFloatArg(ArgKind kind, const Any& arg, std::vector<KernelArgSlot>* slots, std::vector<void*>* out) {
  double value = arg.cast<double>();
  slots->emplace_back();
  KernelArgSlot& slot = slots->back();
  switch (kind) {
    case ArgKind::kFp16:
      slot.fp16 = __float2half(static_cast<float>(value));
      out->push_back(&slot.fp16);
      break;
    case ArgKind::kBf16:
      slot.bf16 = __float2bfloat16(static_cast<float>(value));
      out->push_back(&slot.bf16);
      break;
    case ArgKind::kFp32:
      slot.fp32 = static_cast<float>(value);
      out->push_back(&slot.fp32);
      break;
    case ArgKind::kFp64:
      slot.fp64 = value;
      out->push_back(&slot.fp64);
      break;
    default:
      TVM_FFI_THROW(RuntimeError) << "Invalid float arg kind";
  }
}

template <typename ArgAccess>
inline void LaunchPackedImpl(int64_t registry_handle,
                             int32_t grid_x,
                             int32_t grid_y,
                             int32_t grid_z,
                             ArgAccess&& get_arg,
                             int32_t num_args,
                             Any* ret) {
  RegisteredKernel* kernel = GetRegisteredKernel(registry_handle);

  DLDevice device{};
  bool device_initialized = false;
  auto bind_device = [&](DLDevice candidate, const char* arg_name) {
    TVM_FFI_CHECK(candidate.device_type == kDLCUDA, ValueError)
        << "TVM-FFI Triton export only supports CUDA tensors, got device_type="
        << candidate.device_type << " for argument " << arg_name;
    if (!device_initialized) {
      device = candidate;
      device_initialized = true;
      return;
    }
    TVM_FFI_CHECK(device.device_type == candidate.device_type && device.device_id == candidate.device_id, ValueError)
        << "All tensor arguments must live on the same CUDA device.";
  };

  constexpr size_t kSmallBuf = 32;
  std::array<KernelArgSlot, kSmallBuf> slots_buf;
  std::array<void*, kSmallBuf> launch_buf;
  std::vector<KernelArgSlot> slots_heap;
  std::vector<void*> launch_heap;

  KernelArgSlot* slots = slots_buf.data();
  void** launch_args = launch_buf.data();
  size_t slot_capacity = kSmallBuf;
  size_t launch_capacity = kSmallBuf;
  size_t slot_idx = 0;
  size_t launch_idx = 0;

  auto push_slot = [&]() -> KernelArgSlot& {
    if (slot_idx >= slot_capacity) {
      if (slots_heap.empty()) {
        slots_heap.reserve(kSmallBuf * 2);
        slots_heap.insert(slots_heap.end(), slots_buf.begin(), slots_buf.end());
      } else {
        slots_heap.reserve(slot_capacity * 2);
      }
      slots_heap.resize(slot_idx + 1);
      slots = slots_heap.data();
      slot_capacity = slots_heap.capacity();
    }
    return slots[slot_idx++];
  };

  auto push_launch = [&](void* ptr) {
    if (launch_idx >= launch_capacity) {
      if (launch_heap.empty()) {
        launch_heap.reserve(kSmallBuf * 2);
        launch_heap.insert(launch_heap.end(), launch_buf.begin(), launch_buf.end());
      } else {
        launch_heap.reserve(launch_capacity * 2);
      }
      launch_heap.resize(launch_idx + 1);
      launch_args = launch_heap.data();
      launch_capacity = launch_heap.capacity();
    }
    launch_args[launch_idx++] = ptr;
  };

  size_t arg_index = 0;

  for (const RuntimeArgSpec& spec : kernel->runtime_args) {
    switch (spec.kind) {
      case ArgKind::kPointer: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        TensorView tensor = get_arg(static_cast<int64_t>(arg_index++)).template cast<TensorView>();
        bind_device(tensor.device(), spec.name.c_str());
        KernelArgSlot& slot = push_slot();
        slot.ptr = tensor.data_ptr();
        push_launch(&slot.ptr);
        break;
      }
      case ArgKind::kI1: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        KernelArgSlot& slot = push_slot(); slot.b = get_arg(static_cast<int64_t>(arg_index++)).template cast<bool>(); push_launch(&slot.b); break;
      }
      case ArgKind::kI8: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        KernelArgSlot& slot = push_slot(); slot.i8 = get_arg(static_cast<int64_t>(arg_index++)).template cast<int8_t>(); push_launch(&slot.i8); break;
      }
      case ArgKind::kI16: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        KernelArgSlot& slot = push_slot(); slot.i16 = get_arg(static_cast<int64_t>(arg_index++)).template cast<int16_t>(); push_launch(&slot.i16); break;
      }
      case ArgKind::kI32: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        KernelArgSlot& slot = push_slot(); slot.i32 = get_arg(static_cast<int64_t>(arg_index++)).template cast<int32_t>(); push_launch(&slot.i32); break;
      }
      case ArgKind::kI64: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        KernelArgSlot& slot = push_slot(); slot.i64 = get_arg(static_cast<int64_t>(arg_index++)).template cast<int64_t>(); push_launch(&slot.i64); break;
      }
      case ArgKind::kU1: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        KernelArgSlot& slot = push_slot(); slot.b = get_arg(static_cast<int64_t>(arg_index++)).template cast<bool>(); push_launch(&slot.b); break;
      }
      case ArgKind::kU8: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        KernelArgSlot& slot = push_slot(); slot.u8 = get_arg(static_cast<int64_t>(arg_index++)).template cast<uint8_t>(); push_launch(&slot.u8); break;
      }
      case ArgKind::kU16: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        KernelArgSlot& slot = push_slot(); slot.u16 = get_arg(static_cast<int64_t>(arg_index++)).template cast<uint16_t>(); push_launch(&slot.u16); break;
      }
      case ArgKind::kU32: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        KernelArgSlot& slot = push_slot(); slot.u32 = get_arg(static_cast<int64_t>(arg_index++)).template cast<uint32_t>(); push_launch(&slot.u32); break;
      }
      case ArgKind::kU64: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        KernelArgSlot& slot = push_slot(); slot.u64 = get_arg(static_cast<int64_t>(arg_index++)).template cast<uint64_t>(); push_launch(&slot.u64); break;
      }
      case ArgKind::kFp16: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        KernelArgSlot& slot = push_slot(); slot.fp16 = __float2half(static_cast<float>(get_arg(static_cast<int64_t>(arg_index++)).template cast<double>())); push_launch(&slot.fp16); break;
      }
      case ArgKind::kBf16: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        KernelArgSlot& slot = push_slot(); slot.bf16 = __float2bfloat16(static_cast<float>(get_arg(static_cast<int64_t>(arg_index++)).template cast<double>())); push_launch(&slot.bf16); break;
      }
      case ArgKind::kFp32: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        KernelArgSlot& slot = push_slot(); slot.fp32 = static_cast<float>(get_arg(static_cast<int64_t>(arg_index++)).template cast<double>()); push_launch(&slot.fp32); break;
      }
      case ArgKind::kFp64: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        KernelArgSlot& slot = push_slot(); slot.fp64 = get_arg(static_cast<int64_t>(arg_index++)).template cast<double>(); push_launch(&slot.fp64); break;
      }
      case ArgKind::kNvTmaDesc: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        int64_t ptr_val = get_arg(static_cast<int64_t>(arg_index++)).template cast<int64_t>();
        KernelArgSlot& slot = push_slot();
        slot.ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr_val));
        push_launch(&slot.ptr);
        break;
      }
      case ArgKind::kTensorDesc: {
        TVM_FFI_CHECK(arg_index + static_cast<size_t>(2 * spec.rank + 2) <= static_cast<size_t>(num_args), ValueError)
            << "Missing tensor descriptor runtime arguments for " << spec.name;
        TensorView base = get_arg(static_cast<int64_t>(arg_index++)).template cast<TensorView>();
        bind_device(base.device(), spec.name.c_str());

        std::array<int64_t, 5> shape = {0, 0, 0, 0, 0};
        std::array<int64_t, 5> stride = {0, 0, 0, 0, 0};
        for (int i = 0; i < spec.rank; ++i) {
          int64_t dim = get_arg(static_cast<int64_t>(arg_index++)).template cast<int64_t>();
          TVM_FFI_CHECK(dim >= 0 && dim <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()), ValueError)
              << "Tensor descriptor " << spec.name << " shape[" << i << "] must fit in int32.";
          shape[static_cast<size_t>(i)] = dim;
        }
        for (int i = 0; i < spec.rank; ++i) {
          stride[static_cast<size_t>(i)] = get_arg(static_cast<int64_t>(arg_index++)).template cast<int64_t>();
        }
        bool padding_nan = get_arg(static_cast<int64_t>(arg_index++)).template cast<bool>();

        if (spec.tensordesc_meta.present) {
          KernelArgSlot& slot = push_slot();
          FillTmaDescriptor(&slot.tensor_map,
                            base.data_ptr(),
                            spec.tensordesc_meta.swizzle,
                            spec.tensordesc_meta.elem_size,
                            spec.tensordesc_meta.elem_type,
                            spec.tensordesc_meta.block_size.data(),
                            spec.rank,
                            shape.data(),
                            stride.data(),
                            padding_nan,
                            spec.tensordesc_meta.fp4_padded,
                            spec.name.c_str());
          push_launch(&slot.tensor_map);
          for (int i = 0; i < spec.rank; ++i) {
            KernelArgSlot& s = push_slot();
            s.i32 = static_cast<int32_t>(shape[static_cast<size_t>(i)]);
            push_launch(&s.i32);
          }
          for (int i = 0; i < spec.rank; ++i) {
            KernelArgSlot& s = push_slot();
            s.i64 = stride[static_cast<size_t>(i)];
            push_launch(&s.i64);
          }
        } else {
          KernelArgSlot& slot = push_slot();
          slot.ptr = base.data_ptr();
          push_launch(&slot.ptr);
          for (int i = 0; i < spec.rank; ++i) {
            KernelArgSlot& s = push_slot();
            s.i64 = shape[static_cast<size_t>(i)];
            push_launch(&s.i64);
          }
          for (int i = 0; i < spec.rank; ++i) {
            KernelArgSlot& s = push_slot();
            s.i64 = stride[static_cast<size_t>(i)];
            push_launch(&s.i64);
          }
          KernelArgSlot& pad_slot = push_slot();
          pad_slot.b = padding_nan;
          push_launch(&pad_slot.b);
          for (int i = 0; i < spec.rank; ++i) {
            KernelArgSlot& s = push_slot();
            s.i32 = static_cast<int32_t>(shape[static_cast<size_t>(i)]);
            push_launch(&s.i32);
          }
          for (int i = 0; i < spec.rank; ++i) {
            KernelArgSlot& s = push_slot();
            s.i64 = stride[static_cast<size_t>(i)];
            push_launch(&s.i64);
          }
        }
        break;
      }
    }
  }

  TVM_FFI_CHECK(arg_index == static_cast<size_t>(num_args), ValueError)
      << "Unexpected extra launch arguments: got " << num_args << ", consumed " << arg_index;

  if (!device_initialized) {
    int current_device = 0;
    TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaGetDevice(&current_device));
    device.device_type = kDLCUDA;
    device.device_id = current_device;
    device_initialized = true;
  }

  int previous_device = -1;
  TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaGetDevice(&previous_device));
  if (previous_device != device.device_id) {
    TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaSetDevice(device.device_id));
  }

  tvm::ffi::dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y), static_cast<unsigned>(grid_z));
  tvm::ffi::dim3 block(kernel->block_x, 1u, 1u);
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));

  void* global_scratch = GetScratchBuffer(
      g_global_scratch_buffers,
      device.device_id,
      static_cast<size_t>(grid_x) * static_cast<size_t>(grid_y) * static_cast<size_t>(grid_z) *
          kernel->global_scratch_size,
      kernel->global_scratch_align);
  void* profile_scratch = GetScratchBuffer(
      g_profile_scratch_buffers,
      device.device_id,
      static_cast<size_t>(grid_x) * static_cast<size_t>(grid_y) * static_cast<size_t>(grid_z) *
          kernel->profile_scratch_size,
      kernel->profile_scratch_align);

  push_launch(&global_scratch);
  push_launch(&profile_scratch);

  TVM_FFI_CHECK(kernel->function != nullptr, RuntimeError) << "RegisteredKernel has no CUfunction";
  if (grid_x > 0 && grid_y > 0 && grid_z > 0) {
    CUresult result;
    {
      PyGILState_STATE gil_state = PyGILState_Ensure();
      Py_BEGIN_ALLOW_THREADS;
      result = cuLaunchKernel(kernel->function,
                              grid.x, grid.y, grid.z,
                              block.x, block.y, block.z,
                              kernel->shared_memory,
                              stream,
                              launch_args,
                              nullptr);
      Py_END_ALLOW_THREADS;
      PyGILState_Release(gil_state);
    }
    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(result);
  }

  if (previous_device != device.device_id) {
    PyGILState_STATE gil_state = PyGILState_Ensure();
    Py_BEGIN_ALLOW_THREADS;
    TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaSetDevice(previous_device));
    Py_END_ALLOW_THREADS;
    PyGILState_Release(gil_state);
  }
  *ret = nullptr;
}

inline void LaunchPacked(PackedArgs args, Any* ret) {
  TVM_FFI_CHECK(args.size() >= 4, ValueError) << "Expected at least 4 launch arguments, got " << args.size();
  int64_t registry_handle = args[0].cast<int64_t>();
  int32_t grid_x = args[1].cast<int32_t>();
  int32_t grid_y = args[2].cast<int32_t>();
  int32_t grid_z = args[3].cast<int32_t>();
  auto get_arg = [&](int64_t idx) { return args[static_cast<int32_t>(idx + 4)]; };
  LaunchPackedImpl(registry_handle, grid_x, grid_y, grid_z, get_arg, args.size() - 4, ret);
}

}  // namespace triton_runner_tvm_ffi

TVM_FFI_DLL_EXPORT_TYPED_FUNC(register_kernel_from_function, triton_runner_tvm_ffi::RegisterKernelFromFunction)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_bound_args_launcher, triton_runner_tvm_ffi::MakeBoundArgsLauncher)

extern "C" TVM_FFI_DLL_EXPORT int __tvm_ffi_launch(void* self,
                                                   const TVMFFIAny* args,
                                                   int32_t num_args,
                                                   TVMFFIAny* result) {
  TVM_FFI_SAFE_CALL_BEGIN();
  triton_runner_tvm_ffi::LaunchPacked(
      tvm::ffi::PackedArgs(reinterpret_cast<const tvm::ffi::AnyView*>(args), num_args),
      reinterpret_cast<tvm::ffi::Any*>(result));
  TVM_FFI_SAFE_CALL_END();
}
