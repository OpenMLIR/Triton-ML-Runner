import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from triton.runtime import driver
from triton.runtime.cache import get_cache_manager, get_dump_manager, get_override_manager
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import make_backend, parse, filter_traceback
from triton.compiler.compiler import ASTSource, IRSource
from triton._C.libtriton import get_cache_invalidating_env_vars, ir, llvm

from .checks import runner_check_triton
from ..debug.console import print_triton_cache_dir
from ..compat.triton import triton_key
from .. import __version__
from triton.compiler.compiler import CompiledKernel
from .pass_pipeline import build_pipeline_for_stage
from .source_types import (
    AMD_IR_SOURCE_EXTS,
    BINARY_SOURCE_EXTS,
    NV_IR_SOURCE_EXTS,
    PRECOMPILED_SOURCE_EXTS,
    STORE_ONLY_BINARY_EXTS,
    TEXT_FILE_SOURCE_EXTS,
)
from ..compat.version import is_triton_v3_4, is_disable_multithreading
from ..compat.version import is_tlx_v3_7_4, is_triton_leq_v3_2, is_triton_leq_v3_1, is_triton_geq_v3_5
from ..compat.version import triton_version


def _get_path_ext(src):
    return Path(src).suffix[1:]


def _get_src_ext(src):
    return src.ext if isinstance(src, (ASTSource, IRSource)) else _get_path_ext(src)


def _get_src_hash(src, module):
    if isinstance(src, (ASTSource, IRSource)):
        return src.hash()
    if _get_path_ext(src) in BINARY_SOURCE_EXTS:
        return hashlib.sha256(module).hexdigest()
    return hashlib.sha256(module.encode("utf-8")).hexdigest()



def _load_ir_source_module(src, context, backend):
    """Read an IR file from disk, returning (src, module).

    For plain text IR formats, src is replaced with an IRSource wrapper and
    module is None. For binary formats (cubin/hsaco), module holds the raw bytes.
    """
    assert isinstance(src, str), "source must be either AST or a filepath"
    module = None
    src_ext = _get_path_ext(src)
    if src_ext in TEXT_FILE_SOURCE_EXTS:
        llvm.init_targets()
        module = Path(src).read_text()
    elif src_ext in BINARY_SOURCE_EXTS:
        module = Path(src).read_bytes()
    else:
        src = IRSource(src) if is_triton_leq_v3_2 else IRSource(src, context, backend)
    return src, module


def _build_initial_metadata(kernel_signature, hash, target, options, env_vars):
    """Construct the initial metadata dict for a compilation run."""
    metadata = {
        "kernel_signature": str(kernel_signature),
        "hash": hash,
        "target": target,
        **options.__dict__,
        **env_vars,
    }
    metadata["triton_version"] = triton_version
    metadata["triton_runner_version"] = __version__
    return metadata


def _merge_metadata_from_json(metadata, metadata_json):
    """Copy kernel-specific fields from metadata_json into the compilation metadata dict."""
    metadata["name"] = metadata_json["name"]
    metadata["shared"] = metadata_json["shared"]
    if not is_triton_leq_v3_2:
        metadata["kernel_signature"] = metadata_json.get("kernel_signature", None)
        metadata["cluster_dims"] = metadata_json.get("cluster_dims", (1, 1, 1))
        metadata["tensordesc_meta"] = metadata_json.get("tensordesc_meta", None)
        metadata["num_warps"] = metadata_json.get("num_warps", 4)
        metadata["tmem_size"] = metadata_json.get("tmem_size", 0)
        metadata["global_scratch_size"] = metadata_json.get("global_scratch_size", 0)
        metadata["global_scratch_align"] = metadata_json.get("global_scratch_align", 1)
        metadata["profile_scratch_size"] = metadata_json.get("profile_scratch_size", 0)
        metadata["profile_scratch_align"] = metadata_json.get("profile_scratch_align", 1)


def _llvm_convert_and_optimize(mod, metadata, options, capability, backend):
    """Convert MLIR module to LLVM IR, optimize, and extract metadata."""
    from triton.backends.nvidia.compiler import get_features, sm_arch_from_capability
    from triton import knobs
    from triton._C.libtriton import nvidia

    llvm.init_targets()
    context = llvm.context()
    if knobs.compilation.enable_asan:
        raise RuntimeError(
            "Address Sanitizer Error: Address sanitizer is currently only supported on the AMD backend")
    llvm_mod = llvm.to_module(mod, context)
    proc = sm_arch_from_capability(capability)
    features = get_features(options, getattr(backend.target, 'arch', capability))
    triple = 'nvptx64-nvidia-cuda'
    nvidia.set_short_ptr()
    llvm.attach_datalayout(llvm_mod, triple, proc, features)
    nvidia.set_nvvm_reflect_ftz(llvm_mod)

    if options.extern_libs:
        paths = [path for (name, path) in options.extern_libs]
        llvm.link_extern_libs(llvm_mod, paths)

    llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

    total_num_warps = mod.get_int_attr("ttg.total-num-warps")
    if total_num_warps is not None:
        metadata["num_warps"] = total_num_warps
    metadata["shared"] = mod.get_int_attr("ttg.shared")
    metadata["tmem_size"] = mod.get_int_attr("ttg.tensor_memory_size")
    metadata["global_scratch_size"] = mod.get_int_attr("ttg.global_scratch_memory_size")
    metadata["global_scratch_align"] = mod.get_int_attr("ttg.global_scratch_memory_alignment")
    ret = str(llvm_mod)
    del llvm_mod
    del context
    return ret


def native_compile(src, ast_src, metadata_json=dict(), target=None, options=None, kernel_signature=None, source_path=None, start_pass=None):
    if target is None:
        target = driver.active.get_current_target()
    assert isinstance(target, GPUTarget), "target must be of GPUTarget type"
    backend = make_backend(target)
    ir_source = not isinstance(src, ASTSource)
    module = None
    if ir_source:
        context = ir.context()
        src, module = _load_ir_source_module(src, context, backend)

    ast_extra_options = ast_src.parse_options()
    extra_options = src.parse_options() if isinstance(src, (ASTSource, IRSource)) else {}
    # merge dictionaries, with ast_extra_options(your python code) having higher priority
    extra_options = extra_options | ast_extra_options
    options = backend.parse_options(dict(options or dict(), **extra_options))

    # create cache manager
    env_vars = get_cache_invalidating_env_vars()
    src_hash = _get_src_hash(src, module)
    key = get_cache_key(src_hash, backend, options, env_vars=env_vars)
    hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    fn_cache_manager = get_cache_manager(hash)
    # For dumping/overriding only hash the source as we want it to be independent of triton
    # core changes to make it easier to track kernels by hash.
    enable_override = os.environ.get("TRITON_KERNEL_OVERRIDE", "0") == "1"
    enable_ir_dump = os.environ.get("TRITON_KERNEL_DUMP", "0") == "1"
    store_only_binary = os.environ.get("TRITON_STORE_BINARY_ONLY", "0") == "1"
    fn_override_manager = get_override_manager(src_hash) if enable_override else None
    fn_dump_manager = get_dump_manager(src_hash) if enable_ir_dump else None
    # Pre-truncate the file name here to avoid hitting the 255 character limit on common platforms.
    # The final file name in the cache will have a format of f"{filename}.{ext}.tmp.pid_{pid}_{uuid}".
    # A PID string can be 5-character long. A UUID string has typically 36 characters. Let's truncate
    # the file name to 150 characters to be safe.
    file_name = ast_src.name[:150]
    if not isinstance(src, ASTSource):
        file_name = "@" + ast_src.name
    if metadata_json:
        runner_check_triton(ast_src.name[:150], metadata_json, target)
    metadata_filename = f"{file_name}.json"
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
    metadata_path = metadata_group.get(metadata_filename)

    always_compile = os.environ.get("TRITON_ALWAYS_COMPILE", "0") == "1"
    mlir_dump_path = os.environ.get("MLIR_DUMP_PATH", None)
    if not always_compile and metadata_path is not None:
        if mlir_dump_path is None:
            parse_mlir_to_folder(os.path.join(os.path.dirname(metadata_path), "all.mlir"))
        else:
            parse_mlir_to_folder(mlir_dump_path)
        print_triton_cache_dir(metadata_path, cache_hit=True)
        # cache hit!
        return CompiledKernel(ast_src, metadata_group, hash)

    metadata = _build_initial_metadata(kernel_signature, hash, target, options, env_vars)

    # run compilation pipeline and populate metadata
    stages = dict()
    if is_triton_geq_v3_5 or is_tlx_v3_7_4 or is_triton_v3_4:
        if not isinstance(src, str):
            backend.add_stages(stages, options, src.language)
        else:
            from triton.backends.compiler import Language
            backend.add_stages(stages, options, Language.TRITON)
    else:
        backend.add_stages(stages, options)
    src_ext = _get_src_ext(src)
    first_stage = list(stages.keys()).index(src_ext)
    capability = backend._parse_arch(options.arch) if hasattr(backend, '_parse_arch') else None
    ptx_version = None
    if capability is not None:
        from triton.backends.nvidia.compiler import get_ptx_version_from_options
        ptx_version = get_ptx_version_from_options(options, getattr(backend.target, 'arch', capability))

    # when the source is an IR file, don't apply the passes related to this stage. This makes it easier to write IR level tests.
    # TODO: src_ext perhaps don't need in condition, this is source file
    if start_pass and src_ext in ("ttir", "ttgir", "llir"):
        # don't skip the current stage yet — run_from_pass will handle it after module load
        pass
    elif (ir_source and src_ext != "ttir") or (ir_source and is_tlx_v3_7_4):
        first_stage += 1

    # For IRSource, we have already grabbed the context + called both
    # ir.load_dialects and backend.load_dialects.
    if not isinstance(src, IRSource) or src_ext in AMD_IR_SOURCE_EXTS:
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
    if is_triton_leq_v3_2:
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
        codegen_fns = backend.get_codegen_implementation()
    else:
        codegen_fns = backend.get_codegen_implementation(options)
    try:
        if src_ext in NV_IR_SOURCE_EXTS:
            module = src.src
        elif src_ext not in PRECOMPILED_SOURCE_EXTS:
            module = get_module_with_src_with_make_ir(src, backend, target, options, codegen_fns, context)
    except Exception as e:
        filter_traceback(e)
        raise

    if ir_source:
        ir_filename = f"{file_name}.{src_ext}"
        metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)
    else:
        ir_filename = f"{file_name}.source"
        metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)

    if source_path and os.path.exists(source_path):
        filename = os.path.basename(source_path)
        content = f"# {source_path}\n\n{Path(source_path).read_text()}"
        fn_cache_manager.put(content, filename)

    print_triton_cache_dir(metadata_group[ir_filename])

    if mlir_dump_path is None:
        mlir_dump_path = os.path.join(os.path.dirname(metadata_group[ir_filename]), "all.mlir")
    os.environ["MLIR_DUMP_PATH"] = mlir_dump_path

    # --- Pass-level pipeline execution ---
    if start_pass and src_ext in ("ttir", "ttgir", "llir"):
        # LLIR sources loaded from text files are strings; parse as MLIR for the pass pipeline
        if src_ext == "llir" and isinstance(module, str):
            module = ir.parse_mlir_module(src, context)
        pipeline = build_pipeline_for_stage(src_ext, capability, options, ptx_version)
        pass_idx = pipeline.find_pass(start_pass)
        pipeline.run_from(module, metadata, pass_idx + 1, context=context if src_ext == "llir" else None)
        # Extract TTGIR metadata after pass pipeline
        if src_ext == "ttgir":
            metadata["tensordesc_meta"] = module.get_tensordesc_metadata()
        # for LLIR, also run LLVM conversion + optimization
        if src_ext == "llir":
            module = _llvm_convert_and_optimize(module, metadata, options, capability, backend)
        first_stage += 1
        # save IR after pass pipeline execution
        next_ext = list(stages.keys())[first_stage]
        if next_ext == "ptx":
            ir_filename = f"{file_name}.ptx_input"
            metadata_group[ir_filename] = fn_cache_manager.put(module, ir_filename)

    use_ir_loc = os.environ.get("USE_IR_LOC", None)
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        ir_filename = f"{file_name}.{ext}"
        if (fn_override_manager is not None and (full_name := fn_override_manager.get_file(ir_filename)) is not None):
            print(f"\nOverriding kernel with file {full_name}")
            next_module = parse(full_name, ext, context)
        # If TRITON_STORE_BINARY_ONLY is 1, only store binary/json artifacts
        if (not store_only_binary) or (ext in STORE_ONLY_BINARY_EXTS):
            metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
        if fn_dump_manager is not None:
            fn_dump_manager.put(next_module, ir_filename)
        # use an env variable to parse ir from file
        if use_ir_loc == ext:
            ir_full_name = fn_cache_manager.get_file(ir_filename)
            next_module.create_location_snapshot(ir_full_name)
            print(f"Creating new locations for {ir_full_name}")
        module = next_module

    os.environ.pop("MLIR_DUMP_PATH", None)
    parse_mlir_to_folder(mlir_dump_path)

    if metadata_json:
        _merge_metadata_from_json(metadata, metadata_json)

    # write-back metadata
    metadata_group[metadata_filename] = fn_cache_manager.put(json.dumps(metadata, default=vars), metadata_filename,
                                                             binary=False)
    fn_cache_manager.put_group(metadata_filename, metadata_group)

    # Compilation completed, disabling multithreading in context.
    # This is needed to safely finalize threads pool inside context: if current process forks before
    # python GC deletes context object, thread pool in child process will be invalid, which could
    # lead to child crash or hang.
    #
    # However disabling multithreading causes the code to hang if the ASAN pass is enabled
    # this is likely due to the llvm-symbolizer forking a process
    # TODO: Reconcile the difference here between the ASAN and non-ASAN path with enabling
    # multithreading in the MLIR context
    if not os.environ.get("TRITON_ENABLE_ASAN", "0") == "1":
        if is_disable_multithreading:
            context.disable_multithreading()

    return CompiledKernel(ast_src, metadata_group, hash)


def get_module_with_src_with_make_ir(src, backend, target, options, codegen_fns, context):
    if is_triton_leq_v3_1:
        return src.make_ir(options, codegen_fns, context)
    module_map = backend.get_module_map()
    if is_triton_geq_v3_5 or is_tlx_v3_7_4:
        return src.make_ir(target, options, codegen_fns, module_map, context)
    return src.make_ir(options, codegen_fns, module_map, context)


def get_source_ir(src, target=None, options=None):
    if target is None:
        target = driver.active.get_current_target()
    assert isinstance(target, GPUTarget), "target must be of GPUTarget type"
    backend = make_backend(target)

    extra_options = src.parse_options()
    options = backend.parse_options(dict(options or dict(), **extra_options))

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    codegen_fns = backend.get_codegen_implementation(options)
    try:
        module = get_module_with_src_with_make_ir(src, backend, target, options, codegen_fns, context)
    except Exception as e:
        filter_traceback(e)
        raise
    return module


def get_cache_key(src_hash, backend, backend_options, env_vars):
    runner_key = f'{__version__}'
    key = f"{triton_key()}-{runner_key}-{src_hash}-{backend.hash()}-{backend_options.hash()}-{str(sorted(env_vars.items()))}"
    return key


def parse_mlir_to_folder(mlir_path):
    if not os.path.exists(mlir_path) or os.environ.get("MLIR_ENABLE_DUMP", "0") == "0":
        return
    folder_path = os.path.join(os.path.dirname(mlir_path), "mlir")
    shutil.rmtree(folder_path, ignore_errors=True)
    os.makedirs(folder_path, exist_ok=True)
    content = Path(mlir_path).read_text()

    # Upstream MLIR prints "Pass (key) (op)"; fbtriton prints "Pass: key{opts} (op)".
    pattern = re.compile(
        r'// -----// IR Dump Before (?P<pass_name>.*?) '
        r'(?:\((?P<pass_key>.*?)\) )?'
        r'\((?P<operation>.*?)\) //----- //\n'
        r'(?P<body>.*?)(?=// -----// IR Dump Before|\Z)',
        re.DOTALL
    )
    item = 'source'
    title = 'Python ast_to_ttir'
    last_body = None
    idx = -1
    for idx, match in enumerate(pattern.finditer(content)):
        pass_name = match.group("pass_name").strip()
        pass_key = (match.group("pass_key") or "").strip()
        if not pass_key and ":" in pass_name:
            # fbtriton "Pass: key{opts}" form
            pass_name, pass_key = (part.strip() for part in pass_name.split(":", 1))
            pass_key = pass_key.split("{", 1)[0].strip()
        operation = match.group("operation").strip()
        body = match.group("body").strip()
        changed = "-changed" if last_body and last_body != body else ""
        changed_text = ", This Pass IR has changed!\n" if changed else "\n"
        output_path = Path(folder_path) / f"{idx+1:02d}{changed}-{item}.mlir"
        output_path.write_text(f"// IR Dump After {title}{changed_text}// Next run Pass --{pass_key}\n\n{body}")
        item = f"{pass_name}"
        title = f"{item} ({operation})\n// Current Run Pass --{pass_key}"
        last_body = body

    if idx >= 0:
        output_path = Path(folder_path) / f"{idx+2:02d}-{item}.mlir"
        output_path.write_text(f"// IR Dump After {title}\n")
