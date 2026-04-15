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

struct RuntimeArgSpec {
  std::string name;
  ArgKind kind;
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
    TVM_FFI_CHECK(runtime_arg_ranks[i] == 0, ValueError)
        << "Runtime arg " << spec.name << " must have rank 0 (tensor descriptors should be expanded before registration)";
    TVM_FFI_CHECK(runtime_arg_meta_present[i] == 0, ValueError)
        << "Runtime arg " << spec.name << " must not carry tensor metadata";
    TVM_FFI_CHECK(runtime_arg_block_size_offsets[i] == runtime_arg_block_size_offsets[i + 1], ValueError)
        << "Runtime arg " << spec.name << " must not carry block_size values";
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
                             int64_t stream_ptr,
                             int64_t global_scratch_ptr,
                             int64_t profile_scratch_ptr,
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
    TVM_FFI_CHECK(args.size() >= 6, ValueError)
        << "Expected at least 6 launch arguments, got " << args.size();
    TVM_FFI_CHECK(args.size() == plan->total_bound_arg_count + 6, ValueError)
        << "Expected " << plan->total_bound_arg_count << " bound arguments for TVM-FFI launch, got "
        << (args.size() - 6) << ".";

    auto get_arg = [&](int64_t idx) {
      return args[plan->fast_arg_indices[static_cast<size_t>(idx)] + 4];
    };
    LaunchPackedImpl(plan->registry_handle,
                     args[0].cast<int32_t>(),
                     args[1].cast<int32_t>(),
                     args[2].cast<int32_t>(),
                     args[3].cast<int64_t>(),
                     args[args.size() - 2].cast<int64_t>(),
                     args[args.size() - 1].cast<int64_t>(),
                     get_arg,
                     static_cast<int32_t>(plan->fast_arg_indices.size()),
                     ret);
  });
}

template <typename ArgAccess>
inline void LaunchPackedImpl(int64_t registry_handle,
                             int32_t grid_x,
                             int32_t grid_y,
                             int32_t grid_z,
                             int64_t stream_ptr,
                             int64_t global_scratch_ptr,
                             int64_t profile_scratch_ptr,
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
      case ArgKind::kI1:
      case ArgKind::kI8:
      case ArgKind::kI16:
      case ArgKind::kI32:
      case ArgKind::kI64:
      case ArgKind::kU1:
      case ArgKind::kU8:
      case ArgKind::kU16:
      case ArgKind::kU32:
      case ArgKind::kU64: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        KernelArgSlot& slot = push_slot();
        const Any& arg = get_arg(static_cast<int64_t>(arg_index++));
        switch (spec.kind) {
          case ArgKind::kI1:
          case ArgKind::kU1: slot.b = arg.cast<bool>(); push_launch(&slot.b); break;
          case ArgKind::kI8: slot.i8 = arg.cast<int8_t>(); push_launch(&slot.i8); break;
          case ArgKind::kI16: slot.i16 = arg.cast<int16_t>(); push_launch(&slot.i16); break;
          case ArgKind::kI32: slot.i32 = arg.cast<int32_t>(); push_launch(&slot.i32); break;
          case ArgKind::kI64: slot.i64 = arg.cast<int64_t>(); push_launch(&slot.i64); break;
          case ArgKind::kU8: slot.u8 = arg.cast<uint8_t>(); push_launch(&slot.u8); break;
          case ArgKind::kU16: slot.u16 = arg.cast<uint16_t>(); push_launch(&slot.u16); break;
          case ArgKind::kU32: slot.u32 = arg.cast<uint32_t>(); push_launch(&slot.u32); break;
          case ArgKind::kU64: slot.u64 = arg.cast<uint64_t>(); push_launch(&slot.u64); break;
          default: TVM_FFI_THROW(RuntimeError) << "Invalid integer arg kind";
        }
        break;
      }
      case ArgKind::kFp16:
      case ArgKind::kBf16:
      case ArgKind::kFp32:
      case ArgKind::kFp64: {
        TVM_FFI_CHECK(arg_index < static_cast<size_t>(num_args), ValueError)
            << "Missing runtime argument for " << spec.name;
        KernelArgSlot& slot = push_slot();
        double value = get_arg(static_cast<int64_t>(arg_index++)).template cast<double>();
        switch (spec.kind) {
          case ArgKind::kFp16: slot.fp16 = __float2half(static_cast<float>(value)); push_launch(&slot.fp16); break;
          case ArgKind::kBf16: slot.bf16 = __float2bfloat16(static_cast<float>(value)); push_launch(&slot.bf16); break;
          case ArgKind::kFp32: slot.fp32 = static_cast<float>(value); push_launch(&slot.fp32); break;
          case ArgKind::kFp64: slot.fp64 = value; push_launch(&slot.fp64); break;
          default: TVM_FFI_THROW(RuntimeError) << "Invalid float arg kind";
        }
        break;
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
        TVM_FFI_THROW(RuntimeError) << "Tensor descriptor arguments must be expanded before launch; "
                                       "unexpected kTensorDesc in runtime_args for "
                                    << spec.name;
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
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_ptr));

  void* global_scratch = reinterpret_cast<void*>(static_cast<uintptr_t>(global_scratch_ptr));
  void* profile_scratch = reinterpret_cast<void*>(static_cast<uintptr_t>(profile_scratch_ptr));
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
  TVM_FFI_CHECK(args.size() >= 7, ValueError) << "Expected at least 7 launch arguments, got " << args.size();
  int64_t registry_handle = args[0].cast<int64_t>();
  int32_t grid_x = args[1].cast<int32_t>();
  int32_t grid_y = args[2].cast<int32_t>();
  int32_t grid_z = args[3].cast<int32_t>();
  int64_t stream_ptr = args[4].cast<int64_t>();
  int64_t global_scratch_ptr = args[args.size() - 2].cast<int64_t>();
  int64_t profile_scratch_ptr = args[args.size() - 1].cast<int64_t>();
  auto get_arg = [&](int64_t idx) { return args[static_cast<int32_t>(idx + 5)]; };
  LaunchPackedImpl(registry_handle, grid_x, grid_y, grid_z, stream_ptr, global_scratch_ptr, profile_scratch_ptr,
                   get_arg, static_cast<int32_t>(args.size() - 7), ret);
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
