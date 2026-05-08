"""TVM-FFI based CUDA kernel launcher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from triton.runtime import _allocation
from triton.runtime.cache import get_cache_manager

from ._cuda_build import build_module_from_src, cuda_include_dirs, library_dirs

_GENERIC_LAUNCHER_NAME = "triton_runner_tvm_ffi_generic_launcher"
_GENERIC_LAUNCHER_LIBRARIES = ["tvm_ffi", "cudart", "cuda", "dl"]
_registered_kernel_cache: dict[tuple[str, str], int] = {}
_ARG_KIND_CODES = {
    "pointer": 0,
    "i1": 1,
    "i8": 2,
    "i16": 3,
    "i32": 4,
    "i64": 5,
    "u1": 6,
    "u8": 7,
    "u16": 8,
    "u32": 9,
    "u64": 10,
    "fp16": 11,
    "bf16": 12,
    "fp32": 13,
    "f32": 13,
    "fp64": 14,
    "tensordesc": 15,
    "nvTmaDesc": 16,
}


def _get_generic_launcher_source_path() -> Path:
    return Path(__file__).with_name("generic_launcher.cc")


def _build_generic_launcher_module(tvm_ffi_module: Any, module_name: str, build_dir: str | Path, source: str) -> Any:
    from ._support import _shared_library_path

    tvm_ffi_root = Path(tvm_ffi_module.__file__).resolve().parent
    include_dirs = [str(tvm_ffi_root / "include"), *cuda_include_dirs()]
    return build_module_from_src(
        module_name=module_name,
        build_dir=build_dir,
        source=source,
        include_dirs=include_dirs,
        library_dirs=library_dirs(tvm_ffi_root / "lib", include_stubs=True),
        libraries=_GENERIC_LAUNCHER_LIBRARIES,
        ccflags=("-std=c++17",),
        source_ext=".cc",
        final_path=_shared_library_path(build_dir, module_name),
        load_module=lambda path: tvm_ffi_module.load_module(path, keep_module_alive=True),
    )


def _generic_launcher_cache_key(
        *,
        source: str,
        include_dirs: tuple[str, ...] | list[str],
        library_dirs_list: tuple[str, ...] | list[str],
        libraries: tuple[str, ...] | list[str],
        ccflags: tuple[str, ...] | list[str],
) -> str:
    fingerprint = {
        "version": 3,
        "source": source,
        "include_dirs": list(include_dirs),
        "library_dirs": list(library_dirs_list),
        "libraries": list(libraries),
        "ccflags": list(ccflags),
    }
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _registration_token_for_payload(payload: tuple[Any, ...]) -> str:
    hasher = hashlib.sha256()

    def _update(value: Any) -> None:
        if isinstance(value, str):
            hasher.update(b"s\0")
            hasher.update(value.encode("utf-8"))
            hasher.update(b"\0")
            return
        if isinstance(value, int):
            hasher.update(f"i{value}\0".encode("ascii"))
            return
        if isinstance(value, (bytes, bytearray, memoryview)):
            raw = bytes(value)
            hasher.update(b"b\0")
            hasher.update(str(len(raw)).encode("ascii"))
            hasher.update(b"\0")
            hasher.update(raw)
            return
        if isinstance(value, (list, tuple)):
            hasher.update(b"[\0")
            for item in value:
                _update(item)
            hasher.update(b"]\0")
            return
        raise TypeError(f"Unsupported registration payload value: {type(value)!r}")

    _update(payload)
    return hasher.hexdigest()


def _build_registration_payload(artifact: Any) -> tuple[Any, ...]:
    from ._support import (
        _expand_tensordesc_registration_specs,
        _runtime_arg_registration_specs,
        _validate_launch_metadata,
    )

    metadata = artifact.metadata
    _validate_launch_metadata(metadata)
    runtime_entries = tuple(entry for entry in artifact.signature if not entry.is_constexpr)
    runtime_args = _runtime_arg_registration_specs(runtime_entries, metadata)
    runtime_args = _expand_tensordesc_registration_specs(runtime_args)

    runtime_arg_names: list[str] = []
    runtime_arg_kind_codes: list[int] = []
    runtime_arg_ranks: list[int] = []
    runtime_arg_meta_present: list[int] = []
    runtime_arg_swizzles: list[int] = []
    runtime_arg_elem_sizes: list[int] = []
    runtime_arg_elem_types: list[int] = []
    runtime_arg_fp4_padded: list[int] = []
    runtime_arg_block_size_offsets: list[int] = [0]
    runtime_arg_block_size_values: list[int] = []

    for runtime_arg in runtime_args:
        runtime_arg_names.append(str(runtime_arg["name"]))
        kind = str(runtime_arg["kind"])
        runtime_arg_kind_codes.append(_ARG_KIND_CODES[kind])
        runtime_arg_ranks.append(int(runtime_arg.get("rank", 0)))

        metadata_item = runtime_arg.get("metadata")
        if metadata_item is None:
            runtime_arg_meta_present.append(0)
            runtime_arg_swizzles.append(0)
            runtime_arg_elem_sizes.append(0)
            runtime_arg_elem_types.append(0)
            runtime_arg_fp4_padded.append(0)
        else:
            runtime_arg_meta_present.append(1)
            runtime_arg_swizzles.append(int(metadata_item["swizzle"]))
            runtime_arg_elem_sizes.append(int(metadata_item["elem_size"]))
            runtime_arg_elem_types.append(int(metadata_item["elem_type"]))
            runtime_arg_fp4_padded.append(1 if metadata_item.get("fp4_padded") else 0)
            runtime_arg_block_size_values.extend(int(value) for value in metadata_item["block_size"])
        runtime_arg_block_size_offsets.append(len(runtime_arg_block_size_values))

    payload = (
        artifact.function_handle,
        32 * int(metadata["num_warps"]),
        int(metadata.get("shared", 0)),
        int(metadata.get("global_scratch_size", 0)),
        int(metadata.get("global_scratch_align", 1)),
        int(metadata.get("profile_scratch_size", 0)),
        int(metadata.get("profile_scratch_align", 1)),
        runtime_arg_names,
        runtime_arg_kind_codes,
        runtime_arg_ranks,
        runtime_arg_meta_present,
        runtime_arg_swizzles,
        runtime_arg_elem_sizes,
        runtime_arg_elem_types,
        runtime_arg_fp4_padded,
        runtime_arg_block_size_offsets,
        runtime_arg_block_size_values,
    )
    return payload


def _get_or_build_generic_launcher_module() -> tuple[str, Any]:
    from ._support import (
        _require_tvm_ffi,
        _shared_library_path,
    )

    cuda_source = _get_generic_launcher_source_path().read_text()
    tvm_ffi_module, _ = _require_tvm_ffi()
    tvm_ffi_root = Path(tvm_ffi_module.__file__).resolve().parent
    include_dirs = [str(tvm_ffi_root / "include"), *cuda_include_dirs()]
    library_dirs_list = list(library_dirs(tvm_ffi_root / "lib", include_stubs=True))
    ccflags = ("-std=c++17",)
    cache_key = _generic_launcher_cache_key(
        source=cuda_source,
        include_dirs=include_dirs,
        library_dirs_list=library_dirs_list,
        libraries=_GENERIC_LAUNCHER_LIBRARIES,
        ccflags=ccflags,
    )

    module_name = f"{_GENERIC_LAUNCHER_NAME}_{cache_key[:12]}"
    triton_cache_key = hashlib.sha256(f"tvm_ffi_generic_launcher_{cache_key}".encode("utf-8")).hexdigest()
    cache_manager = get_cache_manager(triton_cache_key)
    build_dir = Path(cache_manager.cache_dir)
    lib_path = cache_manager.get_file(_shared_library_path("", module_name).name)

    mod = tvm_ffi_module.load_module(lib_path, keep_module_alive=True) if lib_path is not None else None
    if mod is None:
        mod = _build_generic_launcher_module(tvm_ffi_module, module_name, build_dir, cuda_source)
    return cache_key, mod


def _register_kernel_if_needed(
        launcher_cache_key: str,
        module: Any,
        registration_args: tuple[Any, ...],
) -> int:
    registration_token = _registration_token_for_payload(registration_args)
    cache_key = (launcher_cache_key, registration_token)
    cached = _registered_kernel_cache.get(cache_key)
    if cached is not None:
        return cached
    handle = int(module.register_kernel_from_function(*registration_args))
    _registered_kernel_cache[cache_key] = handle
    return handle


def _lookup_module_function(module: Any, name: str) -> Any:
    if hasattr(module, "__getitem__"):
        try:
            return module[name]
        except Exception:
            pass
    return getattr(module, name)


class TvmFfiLauncher:
    """TVM-FFI kernel launcher with the same call convention as ``CudaLauncher``."""

    def __init__(self, metadata, function_handle):
        from ._support import (
            _CompiledArtifact,
            _expand_tensordesc_signature,
            _get_tensordesc_python_expansion_info,
            _make_bound_args_launcher,
            _parse_kernel_signature,
        )

        metadata_dict = metadata._asdict() if hasattr(metadata, "_asdict") else dict(metadata)
        if function_handle is None:
            raise ValueError("TVM-FFI launcher requires a function handle.")

        kernel_signature = metadata_dict.get("kernel_signature")
        original_signature = _parse_kernel_signature(kernel_signature)
        expanded_signature = _expand_tensordesc_signature(original_signature, metadata_dict)
        self._signature = expanded_signature

        original_runtime_entries = tuple(entry for entry in original_signature if not entry.is_constexpr)
        self._tensordesc_expansion_info = _get_tensordesc_python_expansion_info(
            original_runtime_entries, metadata_dict
        )

        self._global_scratch_size = int(metadata_dict.get("global_scratch_size", 0))
        self._global_scratch_align = int(metadata_dict.get("global_scratch_align", 1))
        self._profile_scratch_size = int(metadata_dict.get("profile_scratch_size", 0))
        self._profile_scratch_align = int(metadata_dict.get("profile_scratch_align", 1))

        launcher_cache_key, self._tvm_mod = _get_or_build_generic_launcher_module()

        artifact = _CompiledArtifact(
            kernel_name=str(metadata_dict["name"]),
            metadata=metadata_dict,
            signature=original_signature,
            function_handle=int(function_handle),
        )
        registration_args = _build_registration_payload(artifact)
        self._registry_handle = _register_kernel_if_needed(
            launcher_cache_key,
            self._tvm_mod,
            registration_args,
        )

        self._tvm_func = _lookup_module_function(self._tvm_mod, "launch")
        self._launch_bound_args_for_tvm_ffi = _make_bound_args_launcher(
            self._tvm_func,
            self._registry_handle,
            expanded_signature,
            tvm_mod=self._tvm_mod,
        )

    def _expand_tensordesc_args(self, args: tuple[Any, ...]) -> tuple[tuple[Any, ...], list[Any]]:
        expansion_info = self._tensordesc_expansion_info
        if not expansion_info:
            return args, []
        args_list = list(args)
        keepalive: list[Any] = []
        for runtime_idx, rank, meta in reversed(expansion_info):
            arg = args_list[runtime_idx]
            base = arg.base
            shape = list(arg.shape)
            stride = list(arg.strides)
            if meta is None:
                padding_nan = bool(getattr(arg, "padding", None) == "nan")
                expanded = [base, *shape, *stride, padding_nan, *shape, *stride]
            else:
                # TMA path: build a CUtensorMap descriptor and pass its in-memory
                # 128-byte struct by value (kernel param is .align 64 .b8 desc[128]).
                # Matches triton.backends.nvidia.driver.make_tensordesc_arg.
                from triton.backends.nvidia.driver import TMA_DTYPE_DEVICE_TO_HOST, TMA_TF32
                import triton as _triton

                elem_type = meta["elem_type"]
                if getattr(arg, "round_f32_to_tf32", False):
                    elem_type = TMA_TF32
                expanded_shape = list(shape)
                if meta.get("fp4_padded"):
                    expanded_shape[-1] *= 2
                padding = 1 if getattr(arg, "padding", None) == "nan" else 0

                cu_tensor_map = _triton.runtime.driver.active.utils.fill_tma_descriptor_tiled(
                    base.data_ptr(),
                    meta["swizzle"],
                    meta["elem_size"],
                    TMA_DTYPE_DEVICE_TO_HOST[elem_type],
                    meta["block_size"],
                    expanded_shape,
                    stride,
                    padding,
                )
                # PyCUtensorMapObject layout: PyObject_HEAD then a 128-aligned
                # CUtensorMap field at the tail. Compute its offset from
                # tp_basicsize so we don't hard-code the alignment padding.
                desc_offset = type(cu_tensor_map).__basicsize__ - 128
                desc_addr = id(cu_tensor_map) + desc_offset
                # Hold the Python descriptor object alive for the duration of
                # the launch — cuLaunchKernel copies the 128 bytes synchronously
                # before returning, so dropping it after launch is safe.
                keepalive.append(cu_tensor_map)
                expanded = [desc_addr, *shape, *stride]
            args_list[runtime_idx:runtime_idx + 1] = expanded
        return tuple(args_list), keepalive

    def launch_metadata(self, grid, stream, *args):
        return None

    def __call__(self, gridX, gridY, gridZ, stream, function, packed_metadata, launch_metadata, launch_enter_hook, launch_exit_hook, *bound_args):
        self.launch(gridX, gridY, gridZ, *bound_args)

    def launch(self, gridX, gridY, gridZ, *args, stream=None):
        if (stream is not None
                and not self._tensordesc_expansion_info
                and self._global_scratch_size == 0
                and self._profile_scratch_size == 0):
            self._launch_bound_args_for_tvm_ffi(gridX, gridY, gridZ, stream, *args, 0, 0)
            return
        from triton.runtime import driver
        if stream is None:
            device = driver.active.get_current_device()
            stream = driver.active.get_current_stream(device)
        if self._tensordesc_expansion_info:
            args, _keepalive = self._expand_tensordesc_args(args)
        grid_size = gridX * gridY * gridZ
        global_scratch = 0
        profile_scratch = 0
        keepalive = []
        if self._global_scratch_size > 0:
            alloc_size = grid_size * self._global_scratch_size
            buf = _allocation._allocator(alloc_size, self._global_scratch_align, stream)
            global_scratch = buf.data_ptr()
            keepalive.append(buf)
        if self._profile_scratch_size > 0:
            alloc_size = grid_size * self._profile_scratch_size
            buf = _allocation._allocator(alloc_size, self._profile_scratch_align, stream)
            profile_scratch = buf.data_ptr()
            keepalive.append(buf)
        self._launch_bound_args_for_tvm_ffi(gridX, gridY, gridZ, stream, *args, global_scratch, profile_scratch)
