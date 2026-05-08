from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FLOAT_SCALAR_KERNEL_TYPES = {
    "fp16": "__half",
    "bf16": "__nv_bfloat16",
    "fp32": "float",
    "f32": "float",
    "fp64": "double",
}

_INTEGER_SCALAR_TYPES = {
    "i1": "bool",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u1": "bool",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
}

_TENSORDESC_PATTERN = re.compile(r"tensordesc<([^[>]*)\[([^]]*)\]>")
_TMA_DTYPE_DEVICE_TO_HOST = {i: i for i in range(16)}
_TMA_DTYPE_DEVICE_TO_HOST[8] = 10
_TMA_DTYPE_DEVICE_TO_HOST[9] = 8
_TMA_DTYPE_DEVICE_TO_HOST[10] = 9
_MAX_TENSOR_DESC_RANK = 5


@dataclass(frozen=True)
class _SignatureEntry:
    name: str
    type_name: str
    specialization: Any
    is_kwargs: bool = False

    @property
    def is_constexpr(self) -> bool:
        return self.type_name == "constexpr"

    @property
    def is_pointer(self) -> bool:
        return self.type_name.startswith("*")

    @property
    def is_float_scalar(self) -> bool:
        return self.type_name in _FLOAT_SCALAR_KERNEL_TYPES

    @property
    def is_integer_scalar(self) -> bool:
        return self.type_name in _INTEGER_SCALAR_TYPES


@dataclass(frozen=True)
class _CompiledArtifact:
    kernel_name: str
    metadata: dict[str, Any]
    signature: tuple[_SignatureEntry, ...]
    function_handle: int = 0


@dataclass(frozen=True)
class _TensorDescSpec:
    entry: _SignatureEntry
    rank: int
    metadata: dict[str, Any] | None

    @property
    def name(self) -> str:
        return self.entry.name


def _require_tvm_ffi():
    try:
        import tvm_ffi  # type: ignore[import-not-found]
        from tvm_ffi import cpp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "triton_runner.tvm_ffi.build_module requires apache-tvm-ffi. "
            "Install it with `pip install triton-runner[tvm-ffi]` or `pip install apache-tvm-ffi`."
        ) from exc
    return tvm_ffi, cpp


def _parse_kernel_signature(kernel_signature: str | None) -> tuple[_SignatureEntry, ...]:
    if kernel_signature in (None, "", "None"):
        raise NotImplementedError("TVM-FFI export requires a concrete Triton kernel_signature.")
    parsed = ast.parse(kernel_signature, mode="eval").body
    if not isinstance(parsed, (ast.List, ast.Tuple)):
        raise TypeError(f"Unsupported kernel_signature shape: {type(parsed)!r}")

    entries: list[_SignatureEntry] = []
    for item in parsed.elts:
        if not isinstance(item, (ast.List, ast.Tuple)):
            raise TypeError(f"Invalid kernel_signature entry: {ast.unparse(item)!r}")
        if len(item.elts) == 4:
            name_node, type_name_node, specialization_node, is_kwargs_node = item.elts
            is_kwargs = ast.literal_eval(is_kwargs_node)
        elif len(item.elts) == 3:
            name_node, type_name_node, specialization_node = item.elts
            is_kwargs = False
        else:
            raise TypeError(
                f"Unsupported kernel_signature entry length {len(item.elts)}: {ast.unparse(item)!r}"
            )

        name = ast.literal_eval(name_node)
        type_name = ast.literal_eval(type_name_node)
        try:
            specialization = ast.literal_eval(specialization_node)
        except (TypeError, ValueError, SyntaxError):
            if type_name != "constexpr":
                raise
            specialization = ast.unparse(specialization_node)

        entries.append(
            _SignatureEntry(
                name=str(name),
                type_name=str(type_name),
                specialization=specialization,
                is_kwargs=bool(is_kwargs),
            ))
    return tuple(entries)


def _shared_library_path(build_dir: str | Path, module_name: str) -> Path:
    ext = ".dll" if os.name == "nt" else ".so"
    return Path(build_dir) / f"{module_name}{ext}"


def _parse_tensordesc_specs(
        runtime_entries: tuple[_SignatureEntry, ...], metadata: dict[str, Any]) -> tuple[_TensorDescSpec, ...]:
    raw_metadata = metadata.get("tensordesc_meta") or ()
    specs: list[_TensorDescSpec] = []
    meta_index = 0

    for entry in runtime_entries:
        if not entry.type_name.startswith("tensordesc"):
            continue

        match = _TENSORDESC_PATTERN.fullmatch(entry.type_name)
        if match is None:
            raise TypeError(f"Unsupported tensordesc Triton signature: {entry.type_name}")

        shape_signature = match.group(2).strip()
        rank = 1 if not shape_signature else shape_signature.count(",") + 1
        if rank > _MAX_TENSOR_DESC_RANK:
            raise NotImplementedError(
                f"TVM-FFI export supports tensor descriptors up to rank {_MAX_TENSOR_DESC_RANK}, got {entry.type_name}."
            )

        spec_metadata: dict[str, Any] | None = None
        if raw_metadata:
            if meta_index >= len(raw_metadata):
                raise ValueError(
                    "Tensor descriptor metadata is shorter than the number of tensordesc runtime arguments."
                )
            raw_item = raw_metadata[meta_index]
            if raw_item is not None:
                block_size = tuple(int(v) for v in raw_item.get("block_size", ()))
                if len(block_size) != rank:
                    raise ValueError(
                        f"Tensor descriptor metadata for {entry.name} has block_size rank {len(block_size)}, expected {rank}."
                    )
                elem_type = int(raw_item["elem_type"])
                spec_metadata = {
                    "swizzle": int(raw_item["swizzle"]),
                    "elem_size": int(raw_item["elem_size"]),
                    "elem_type": int(_TMA_DTYPE_DEVICE_TO_HOST.get(elem_type, elem_type)),
                    "block_size": block_size,
                    "fp4_padded": bool(raw_item.get("fp4_padded", False)),
                }
            meta_index += 1

        specs.append(
            _TensorDescSpec(
                entry=entry,
                rank=rank,
                metadata=spec_metadata,
            ))

    if raw_metadata and specs and meta_index != len(raw_metadata):
        raise ValueError(
            "Tensor descriptor metadata count does not match the number of tensordesc runtime arguments."
        )

    return tuple(specs)


def _make_bound_args_launcher(
        tvm_func: Any,
        registry_handle: int,
        signature: tuple[_SignatureEntry, ...],
        tvm_mod: Any | None = None,
):
    if tvm_mod is None:
        from .driver import _get_or_build_generic_launcher_module

        _, tvm_mod = _get_or_build_generic_launcher_module()
    builder = getattr(tvm_mod, "make_bound_args_launcher")
    return builder(
        tvm_func,
        int(registry_handle),
        [entry.type_name for entry in signature],
    )


def _runtime_arg_registration_specs(
        runtime_entries: tuple[_SignatureEntry, ...],
        metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    descriptor_specs = _parse_tensordesc_specs(runtime_entries, metadata)
    descriptor_index = 0
    runtime_args: list[dict[str, Any]] = []

    for entry in runtime_entries:
        if entry.type_name.startswith("tensordesc"):
            spec = descriptor_specs[descriptor_index]
            descriptor_index += 1
            runtime_args.append({
                "name": spec.name,
                "kind": "tensordesc",
                "rank": spec.rank,
                "metadata": spec.metadata,
            })
            continue

        if entry.is_pointer:
            kind = "pointer"
        elif entry.is_integer_scalar or entry.is_float_scalar:
            kind = entry.type_name
        else:
            raise NotImplementedError(f"Unsupported Triton runtime argument type: {entry.type_name}")

        runtime_args.append({"name": entry.name, "kind": kind})

    return runtime_args


def _expand_tensordesc_registration_specs(
        runtime_args: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for spec in runtime_args:
        if spec.get("kind") != "tensordesc":
            expanded.append(spec)
            continue
        rank = spec["rank"]
        name = spec["name"]
        if spec.get("metadata") is None:
            expanded.append({"name": f"{name}_base", "kind": "pointer"})
            for i in range(rank):
                expanded.append({"name": f"{name}_shape_{i}", "kind": "i64"})
            for i in range(rank):
                expanded.append({"name": f"{name}_stride_{i}", "kind": "i64"})
            expanded.append({"name": f"{name}_padding_nan", "kind": "i1"})
            for i in range(rank):
                expanded.append({"name": f"{name}_shape2_{i}", "kind": "i32"})
            for i in range(rank):
                expanded.append({"name": f"{name}_stride2_{i}", "kind": "i64"})
        else:
            # TMA path: kernel ABI is [CUtensorMap (128B by value), shape*i64, stride*i64].
            # Match what triton.backends.nvidia.driver.make_tensordesc_arg produces.
            expanded.append({"name": f"{name}_tma", "kind": "nvTmaDesc"})
            for i in range(rank):
                expanded.append({"name": f"{name}_shape_{i}", "kind": "i64"})
            for i in range(rank):
                expanded.append({"name": f"{name}_stride_{i}", "kind": "i64"})
    return expanded


def _expand_tensordesc_signature(
        signature: tuple[_SignatureEntry, ...],
        metadata: dict[str, Any],
) -> tuple[_SignatureEntry, ...]:
    runtime_entries = tuple(entry for entry in signature if not entry.is_constexpr)
    descriptor_specs = _parse_tensordesc_specs(runtime_entries, metadata)
    expanded: list[_SignatureEntry] = []
    spec_index = 0
    for entry in signature:
        if entry.is_constexpr:
            expanded.append(entry)
            continue
        if not entry.type_name.startswith("tensordesc"):
            expanded.append(entry)
            continue
        spec = descriptor_specs[spec_index]
        spec_index += 1
        rank = spec.rank
        if spec.metadata is None:
            expanded.append(_SignatureEntry(f"{entry.name}_base", "*", entry.specialization, entry.is_kwargs))
            for i in range(rank):
                expanded.append(_SignatureEntry(f"{entry.name}_shape_{i}", "i64", entry.specialization, entry.is_kwargs))
            for i in range(rank):
                expanded.append(_SignatureEntry(f"{entry.name}_stride_{i}", "i64", entry.specialization, entry.is_kwargs))
            expanded.append(_SignatureEntry(f"{entry.name}_padding_nan", "i1", entry.specialization, entry.is_kwargs))
            for i in range(rank):
                expanded.append(_SignatureEntry(f"{entry.name}_shape2_{i}", "i32", entry.specialization, entry.is_kwargs))
            for i in range(rank):
                expanded.append(_SignatureEntry(f"{entry.name}_stride2_{i}", "i64", entry.specialization, entry.is_kwargs))
        else:
            # TMA path — must match _expand_tensordesc_registration_specs above
            # so the C launcher's argument plan agrees with the kernel ABI.
            expanded.append(_SignatureEntry(f"{entry.name}_tma", "nvTmaDesc", entry.specialization, entry.is_kwargs))
            for i in range(rank):
                expanded.append(_SignatureEntry(f"{entry.name}_shape_{i}", "i64", entry.specialization, entry.is_kwargs))
            for i in range(rank):
                expanded.append(_SignatureEntry(f"{entry.name}_stride_{i}", "i64", entry.specialization, entry.is_kwargs))
    return tuple(expanded)


def _get_tensordesc_python_expansion_info(
        runtime_entries: tuple[_SignatureEntry, ...],
        metadata: dict[str, Any],
) -> list[tuple[int, int, dict[str, Any] | None]]:
    descriptor_specs = _parse_tensordesc_specs(runtime_entries, metadata)
    info: list[tuple[int, int, dict[str, Any] | None]] = []
    spec_index = 0
    runtime_idx = 0
    for entry in runtime_entries:
        if entry.type_name.startswith("tensordesc"):
            spec = descriptor_specs[spec_index]
            spec_index += 1
            info.append((runtime_idx, spec.rank, spec.metadata))
        runtime_idx += 1
    return info


def _validate_launch_metadata(metadata: dict[str, Any]) -> None:
    cluster_dims = tuple(int(v) for v in metadata.get("cluster_dims", (1, 1, 1)))
    cluster_product = cluster_dims[0] * cluster_dims[1] * cluster_dims[2]
    unsupported: list[str] = []
    if cluster_product != 1 or int(metadata.get("num_ctas", 1)) != 1:
        unsupported.append("cluster launch / num_ctas != 1")
    if metadata.get("launch_pdl"):
        unsupported.append("launch_pdl")
    if metadata.get("launch_cooperative_grid"):
        unsupported.append("launch_cooperative_grid")
    if unsupported:
        raise NotImplementedError(
            "TVM-FFI launcher does not support: " + ", ".join(unsupported)
        )
