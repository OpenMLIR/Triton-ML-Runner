import dataclasses
import hashlib
import json
import os

from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction, KernelInterface, T

from ..compiler.compile import native_compile
from ..compiler.source_types import RUNNER_SOURCE_TYPES
from ..compat.triton import get_triton_cache_dir
from .dump import DumpMixin
from .metadata import MetadataMixin


def _normalize_cache_key_value(value):
    if hasattr(value, "_asdict"):
        return _normalize_cache_key_value(value._asdict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _normalize_cache_key_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _normalize_cache_key_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_key_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized_items = [_normalize_cache_key_value(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        )
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        public_attrs = {
            key: attr
            for key, attr in vars(value).items()
            if not key.startswith("_")
        }
        if public_attrs:
            return _normalize_cache_key_value(public_attrs)
    return repr(value)


def _stable_cache_key_digest(value):
    normalized = _normalize_cache_key_value(value)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RunnerJITFunction(DumpMixin, MetadataMixin, JITFunction[KernelInterface[T]]):

    def normalize_runner_kwargs(self, kwargs):
        metadata_json = kwargs.get("metadata_json")
        if hasattr(metadata_json, "_asdict"):
            kwargs["metadata_json"] = metadata_json._asdict()

    def get_cache_key_with_runner_args(self, key, kwargs):
        if kwargs.get("dump_tensor") is not None:
            key += "|dump_tensor"
        if "dump_value" in kwargs:
            key += f"|dump_value={kwargs['dump_value']}"
        if "dump_grid" in kwargs:
            key += f"|dump_grid={kwargs['dump_grid']}"
        if "start_pass" in kwargs:
            key += f"|start_pass={kwargs['start_pass']}"
        if (runner_source_key_suffix := self.get_runner_source_key_suffix(kwargs)):
            key += f"|runner_src={runner_source_key_suffix}"
        if "metadata_json" in kwargs:
            key += f"|runner_metadata={_stable_cache_key_digest(kwargs['metadata_json'])}"
        return key

    def get_runner_args_set(self):
        return RUNNER_SOURCE_TYPES

    def get_dump_args_set(self):
        return {"dump_tensor", "dump_value", "dump_grid"}

    def get_autotune_args_set(self):
        return {"autotune_cubin_dir"}

    def get_metadata_args_set(self):
        return {"metadata_json"}

    @property
    def source_path(self):
        return self.__dict__["__globals__"].get("__file__")

    def get_runner_cache_dir(self):
        runner_cache_dir = os.path.join(get_triton_cache_dir(), "runner_cache")
        os.makedirs(runner_cache_dir, exist_ok=True)
        return runner_cache_dir

    def _check_source_dir_type(self, need_check_lst):
        runner_args_set = self.get_runner_args_set()
        dump_args_set = self.get_dump_args_set()
        autotune_args_set = self.get_autotune_args_set()
        metadata_args_set = self.get_metadata_args_set()
        recognized = runner_args_set | dump_args_set | autotune_args_set | metadata_args_set | {"start_pass"}
        for k in need_check_lst:
            if k not in recognized:
                raise KeyError("Keyword argument %s was specified but unrecognised" % k)
        for k in need_check_lst:
            if k in runner_args_set:
                return k

    def get_source_dir_type(self, kwargs, options, sigkeys):
        return self._check_source_dir_type(
            [k.lower() for k in kwargs if k not in options.__dict__ and k not in sigkeys])

    def get_runner_source_key_suffix(self, kwargs):
        runner_args_set = self.get_runner_args_set()
        for k in kwargs:
            if k in runner_args_set:
                if k.endswith("_src"):
                    src_hash = hashlib.sha256(kwargs[k].encode("utf-8")).hexdigest()
                    return f"{k}:{src_hash}"
                return kwargs[k] + f"/{self.__name__}.{k[:-4]}"
        return ""

    def _pack_args(self, backend, kwargs, bound_args, specialization, options):
        from triton._utils import find_paths_if, get_iterable_path
        # options
        options = backend.parse_options(kwargs)
        # signature
        sigkeys = [x.name for x in self.params]
        sigvals = [x[0] for x in specialization]
        signature = {k: v for (k, v) in zip(sigkeys, sigvals)}
        # check arguments
        assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
        assert "device" not in kwargs, "device option is deprecated; current device will be used"
        assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"
        source_dir_type = self.get_source_dir_type(kwargs, options, sigkeys)
        # constexprs
        constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
        constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}
        # attributes
        attrvals = [x[1] for x in specialization]
        attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
        attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}

        return options, signature, constexprs, attrs, source_dir_type

    def _do_compile(self, key, signature, device, constexprs, options, attrs, warmup, source_dir_type, kwargs, bound_args=None, kernel_signature=None):
        from triton import knobs

        kernel_cache, _, target, backend, _ = self.device_caches[device]

        # [Triton Runner] dump before _call_hook
        src = self.get_src_and_save_dump_file(kwargs, source_dir_type, signature, constexprs, attrs, target, options, bound_args)
        if self.need_dump(kwargs):
            kernel_signature = kernel_signature + (('dump_tensor', '*fp32', 'D', False),)
        if self._call_hook(knobs.runtime.jit_cache_hook, key, signature, device, constexprs, options, [attrs], warmup):
            return None
        ast_src = self.ASTSource(self, signature, constexprs, attrs)

        # [Triton Runner] dump after _call_hook
        src, metadata_json = self.get_src_and_metadata_json(kwargs, source_dir_type, src, ast_src)
        # [Triton Runner] change compile
        kernel = native_compile(src, ast_src, metadata_json, target=target, options=options.__dict__, source_path=self.source_path, kernel_signature=kernel_signature)
        kernel_cache[key] = kernel
        self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, device, constexprs, options, [attrs], warmup)
        return kernel

    def _check_globals(self):
        not_present = object()
        for (name, _), (val, globals_dict) in self.used_global_vals.items():
            if (newVal := globals_dict.get(name, not_present)) != val:
                raise RuntimeError(
                    f"Global variable {name} has changed since we compiled this kernel, from {val} to {newVal}")

    def _resolve_grid(self, grid, bound_args):
        if callable(grid):
            grid = grid(bound_args)
        grid_size = len(grid)
        return grid, grid[0], grid[1] if grid_size > 1 else 1, grid[2] if grid_size > 2 else 1



class RunnerJITFunctionV3_8_0(RunnerJITFunction[KernelInterface[T]]):

    def run(self, *args, grid, warmup, **kwargs):
        self.handle_autotune(kwargs)
        self.normalize_runner_kwargs(kwargs)
        from triton import knobs
        from triton.runtime.jit import compute_cache_key

        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
        kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode

        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
        bound_args, specialization, options = binder(*args, **kwargs)

        if knobs.runtime.add_stages_inspection_hook is not None:
            inspect_stages_key, inspect_stages_hash = knobs.runtime.add_stages_inspection_hook()
            specialization.append(f'("custom_pipeline", {inspect_stages_hash})')

        key = compute_cache_key(kernel_key_cache, specialization, options)
        key = self.get_cache_key_with_runner_args(key, kwargs)
        kernel = kernel_cache.get(key, None)

        if kernel is None:
            options, signature, constexprs, attrs, source_dir_type = self._pack_args(
                backend, kwargs, bound_args, specialization, options)
            kernel_signature = tuple(
                (name, arg_type, spec, name in kwargs)
                for name, (arg_type, spec) in zip(bound_args.keys(), specialization)
            )

            src = self.get_src_and_save_dump_file(
                kwargs, source_dir_type, signature, constexprs, attrs, target, options, bound_args)
            if self.need_dump(kwargs):
                kernel_signature = kernel_signature + (("dump_tensor", "*fp32", "D", False),)

            if JITFunction._call_hook(
                self,
                knobs.runtime.jit_cache_hook,
                key,
                signature,
                target,
                device,
                constexprs,
                options,
                [attrs],
                warmup,
            ):
                return None

            ast_src = self.ASTSource(self, signature, constexprs, attrs)
            src, metadata_json = self.get_src_and_metadata_json(kwargs, source_dir_type, src, ast_src)
            kernel = native_compile(
                src,
                ast_src,
                metadata_json,
                target=target,
                options=options.__dict__,
                source_path=self.source_path,
                kernel_signature=kernel_signature,
            )
            if kernel is None:
                return None
            kernel_cache[key] = kernel
            JITFunction._call_hook(
                self,
                knobs.runtime.jit_post_compile_hook,
                key,
                signature,
                target,
                device,
                constexprs,
                options,
                [attrs],
                warmup,
            )

        self._check_globals()

        if not warmup:
            assert grid is not None
            grid, grid_0, grid_1, grid_2 = self._resolve_grid(grid, bound_args)
            if hasattr(kernel, "result"):
                kernel = kernel.result()
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
        return kernel


class RunnerJITFunctionV3_7_0(RunnerJITFunction[KernelInterface[T]]):

    def run(self, *args, grid, warmup, **kwargs):
        self.handle_autotune(kwargs)
        self.normalize_runner_kwargs(kwargs)
        from triton import knobs
        from triton.runtime.jit import compute_cache_key

        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
        kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode

        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
        bound_args, specialization, options = binder(*args, **kwargs)

        if knobs.runtime.add_stages_inspection_hook is not None:
            inspect_stages_key, inspect_stages_hash = knobs.runtime.add_stages_inspection_hook()
            specialization.append(f'("custom_pipeline", {inspect_stages_hash})')

        hook_key = (tuple(specialization), str(options))
        key = compute_cache_key(kernel_key_cache, specialization, options)
        key = self.get_cache_key_with_runner_args(key, kwargs)
        kernel = kernel_cache.get(key, None)

        if kernel is None:
            options, signature, constexprs, attrs, source_dir_type = self._pack_args(
                backend, kwargs, bound_args, specialization, options)
            kernel_signature = tuple(
                (name, arg_type, spec, name in kwargs)
                for name, (arg_type, spec) in zip(bound_args.keys(), specialization)
            )

            src = self.get_src_and_save_dump_file(
                kwargs, source_dir_type, signature, constexprs, attrs, target, options, bound_args)
            if self.need_dump(kwargs):
                kernel_signature = kernel_signature + (("dump_tensor", "*fp32", "D", False),)

            if JITFunction._call_hook(
                self,
                knobs.runtime.jit_cache_hook,
                hook_key,
                signature,
                target,
                device,
                constexprs,
                options,
                [attrs],
                warmup,
            ):
                return None

            ast_src = self.ASTSource(self, signature, constexprs, attrs)
            src, metadata_json = self.get_src_and_metadata_json(kwargs, source_dir_type, src, ast_src)
            kernel = native_compile(
                src,
                ast_src,
                metadata_json,
                target=target,
                options=options.__dict__,
                source_path=self.source_path,
                kernel_signature=kernel_signature,
            )
            if kernel is None:
                return None
            kernel_cache[key] = kernel
            JITFunction._call_hook(
                self,
                knobs.runtime.jit_post_compile_hook,
                hook_key,
                signature,
                target,
                device,
                constexprs,
                options,
                [attrs],
                warmup,
            )

        self._check_globals()

        if not warmup:
            assert grid is not None
            grid, grid_0, grid_1, grid_2 = self._resolve_grid(grid, bound_args)
            if hasattr(kernel, "result"):
                kernel = kernel.result()
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
        return kernel


class RunnerJITFunctionV3_6_0(RunnerJITFunction[KernelInterface[T]]):

    def run(self, *args, grid, warmup, **kwargs):
        self.handle_autotune(kwargs)
        self.normalize_runner_kwargs(kwargs)
        from triton import knobs
        from triton.runtime.jit import compute_cache_key

        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
        kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode

        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
        bound_args, specialization, options = binder(*args, **kwargs)

        if knobs.runtime.add_stages_inspection_hook is not None:
            inspect_stages_key, inspect_stages_hash = knobs.runtime.add_stages_inspection_hook()
            specialization.append(f'("custom_pipeline", {inspect_stages_hash})')

        key = compute_cache_key(kernel_key_cache, specialization, options)
        key = self.get_cache_key_with_runner_args(key, kwargs)
        kernel = kernel_cache.get(key, None)

        if kernel is None:
            options, signature, constexprs, attrs, source_dir_type = self._pack_args(
                backend, kwargs, bound_args, specialization, options)
            kernel_signature = tuple((key, arg_type, spec, key in kwargs) for key, (arg_type, spec) in zip(bound_args.keys(), specialization))
            kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup, source_dir_type, kwargs, bound_args, kernel_signature)
            if kernel is None:
                return None

        self._check_globals()

        if not warmup:
            assert grid is not None
            grid, grid_0, grid_1, grid_2 = self._resolve_grid(grid, bound_args)
            if hasattr(kernel, "result"):
                kernel = kernel.result()
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
        return kernel


class RunnerJITFunctionV3_5_0(RunnerJITFunction[KernelInterface[T]]):

    def run(self, *args, grid, warmup, **kwargs):
        self.handle_autotune(kwargs)
        self.normalize_runner_kwargs(kwargs)
        from triton import knobs
        from triton.runtime.jit import compute_cache_key

        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug

        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
        bound_args, specialization, options = binder(*args, **kwargs)

        key = compute_cache_key(kernel_key_cache, specialization, options)
        # [Triton Runner] dump key
        key = self.get_cache_key_with_runner_args(key, kwargs)
        kernel = kernel_cache.get(key, None)

        if kernel is None:
            options, signature, constexprs, attrs, source_dir_type = self._pack_args(
                backend, kwargs, bound_args, specialization, options)
            kernel_signature = tuple((key, arg_type, spec, key in kwargs) for key, (arg_type, spec) in zip(bound_args.keys(), specialization))
            kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup, source_dir_type, kwargs, bound_args, kernel_signature)
            if kernel is None:
                return None

        self._check_globals()

        if not warmup:
            assert grid is not None
            grid, grid_0, grid_1, grid_2 = self._resolve_grid(grid, bound_args)
            if hasattr(kernel, "result"):
                kernel = kernel.result()
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
        return kernel


class RunnerJITFunctionV3_4_0(RunnerJITFunction[KernelInterface[T]]):

    def run(self, *args, grid, warmup, **kwargs):
        from triton import knobs
        self.handle_autotune(kwargs)
        self.normalize_runner_kwargs(kwargs)

        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug

        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, target, backend, binder = self.device_caches[device]
        bound_args, specialization, options = binder(*args, **kwargs)

        key = str(specialization) + str(options)
        # [Triton Runner] dump key
        key = self.get_cache_key_with_runner_args(key, kwargs)
        kernel = kernel_cache.get(key, None)

        if kernel is None:
            options, signature, constexprs, attrs, source_dir_type = self._pack_args(
                backend, kwargs, bound_args, specialization, options)

            # [Triton Runner] dump before _call_hook
            src = self.get_src_and_save_dump_file(kwargs, source_dir_type, signature, constexprs, attrs, target, options, bound_args)
            if self._call_hook(knobs.runtime.jit_cache_hook, key, signature, device, constexprs, options, [attrs], warmup):
                return None
            ast_src = self.ASTSource(self, signature, constexprs, attrs)
            # [Triton Runner] dump after _call_hook
            src, metadata_json = self.get_src_and_metadata_json(kwargs, source_dir_type, src, ast_src)
            kernel_signature = tuple((key, arg_type, spec) for key, (arg_type, spec) in zip(bound_args.keys(), specialization))
            kernel = native_compile(src, ast_src, metadata_json, target=target, options=options.__dict__, source_path=self.source_path, kernel_signature=kernel_signature, start_pass=kwargs.get("start_pass"))
            kernel_cache[key] = kernel
            self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, device, constexprs, options, [attrs], warmup)

        self._check_globals()

        if not warmup:
            assert grid is not None
            grid, grid_0, grid_1, grid_2 = self._resolve_grid(grid, bound_args)
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
        return kernel


class RunnerJITFunction_TLX(RunnerJITFunction[KernelInterface[T]]):

    def run(self, *args, grid, warmup, **kwargs):
        from triton import knobs
        from triton.runtime.jit import compute_cache_key
        self.normalize_runner_kwargs(kwargs)

        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug

        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
        bound_args, specialization, options = binder(*args, **kwargs)

        key = compute_cache_key(kernel_key_cache, specialization, options)
        key = self.get_cache_key_with_runner_args(key, kwargs)
        kernel = kernel_cache.get(key, None)

        if kernel is None:
            options, signature, constexprs, attrs, source_dir_type = self._pack_args(
                backend, kwargs, bound_args, specialization, options)
            kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup, source_dir_type, kwargs)
            if kernel is None:
                return None

        self._check_globals()

        if not warmup:
            assert grid is not None
            grid, grid_0, grid_1, grid_2 = self._resolve_grid(grid, bound_args)
            if hasattr(kernel, "result"):
                kernel = kernel.result()
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
        return kernel


class RunnerJITFunctionV3_3_0(RunnerJITFunction[KernelInterface[T]]):

    def run(self, *args, grid, warmup, **kwargs):
        self.normalize_runner_kwargs(kwargs)
        kwargs["debug"] = kwargs.get("debug", self.debug) or os.environ.get("TRITON_DEBUG", "0") == "1"

        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, target, backend, binder = self.device_caches[device]
        bound_args, specialization, options = binder(*args, **kwargs)

        key = str(specialization) + str(options)
        # [Triton Runner] dump key
        key = self.get_cache_key_with_runner_args(key, kwargs)
        kernel = kernel_cache.get(key, None)

        if kernel is None:
            options, signature, constexprs, attrs, source_dir_type = self._pack_args(
                backend, kwargs, bound_args, specialization, options)

            # [Triton Runner] dump before _call_hook
            src = self.get_src_and_save_dump_file(kwargs, source_dir_type, signature, constexprs, attrs, target, options, bound_args)
            if self._call_hook(key, signature, device, constexprs, options, [attrs], warmup, before=True):
                return None
            ast_src = self.ASTSource(self, signature, constexprs, attrs)
            # [Triton Runner] dump after _call_hook
            src, metadata_json = self.get_src_and_metadata_json(kwargs, source_dir_type, src, ast_src)
            kernel_signature = tuple((key, arg_type, spec) for key, (arg_type, spec) in zip(bound_args.keys(), specialization))
            kernel = native_compile(src, ast_src, metadata_json, target=target, options=options.__dict__, source_path=self.source_path, kernel_signature=kernel_signature)
            kernel_cache[key] = kernel
            self._call_hook(key, signature, device, constexprs, options, [attrs], warmup, before=False)

        self._check_globals()

        if not warmup:
            assert grid is not None
            grid, grid_0, grid_1, grid_2 = self._resolve_grid(grid, bound_args)
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       self.CompiledKernel.launch_enter_hook, self.CompiledKernel.launch_exit_hook, *bound_args.values())
        return kernel


class RunnerJITFunctionV3_2_0(RunnerJITFunction[KernelInterface[T]]):

    def get_source_dir_type(self, excess_kwargs, options):
        return self._check_source_dir_type([k.lower() for k in excess_kwargs if k not in options.__dict__])

    def run(self, *args, grid, warmup, **kwargs):
        self.normalize_runner_kwargs(kwargs)
        kwargs["debug"] = kwargs.get("debug", False) or os.environ.get("TRITON_DEBUG", "0") == "1"

        from triton.compiler import make_backend
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        target = driver.active.get_current_target()
        backend = make_backend(target)

        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        if self.binder is None:
            self.create_binder(backend)

        bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)

        key = ''.join(sig_and_spec) + str((constexpr_vals, excess_kwargs))
        kernel = self.cache[device].get(key, None)

        if kernel is None:
            options = backend.parse_options(kwargs)

            assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
            assert "device" not in kwargs, "device option is deprecated; current device will be used"
            assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"

            source_dir_type = self.get_source_dir_type(excess_kwargs, options)

            bound_vals = tuple(bound_args.values())

            sigkeys = [self.params[i].name for i in self.non_constexpr_indices]
            sigvals = sig_and_spec[:len(sigkeys)]
            signature = {k: ('*i8' if (v == 'none') else v) for (k, v) in zip(sigkeys, sigvals)}

            configs = (backend.get_attrs_descriptor(self.params, bound_vals), )
            constant_params = configs[0].get_constants()
            constants = {
                p.name: v
                for (v, p) in zip(bound_vals, self.params)
                if p.is_constexpr or (p.num in constant_params) or v is None
            }
            for i, arg in constants.items():
                if callable(arg):
                    raise TypeError(f"Callable constexpr at index {i} is not supported")

            if self._call_hook(key, signature, device, constants, options, configs, warmup, before=True):
                return None
            ast_src = self.ASTSource(self, signature, constants, configs[0])
            src, metadata_json = self.get_src_and_metadata_json(kwargs, source_dir_type, None, ast_src)
            kernel = native_compile(src, ast_src, metadata_json, target=target, options=options.__dict__, source_path=self.source_path)
            self.cache[device][key] = kernel
            self._call_hook(key, signature, device, constants, options, configs, warmup, before=False)

        # Check that used global values have not changed.
        not_present = object()
        for (name, _), (val, globals_dict) in self.used_global_vals.items():
            if (newVal := globals_dict.get(name, not_present)) != val:
                raise RuntimeError(
                    f"Global variable {name} has changed since we compiled this kernel, from {val} to {newVal}")

        if not warmup:
            assert grid is not None
            grid, grid_0, grid_1, grid_2 = self._resolve_grid(grid, bound_args)
            launch_metadata = kernel.launch_metadata(grid, stream, *non_constexpr_vals)
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       self.CompiledKernel.launch_enter_hook, self.CompiledKernel.launch_exit_hook, *non_constexpr_vals)
        return kernel


class RunnerJITFunctionV3_1_0(RunnerJITFunction[KernelInterface[T]]):

    def get_source_dir_type(self, excess_kwargs, options):
        return self._check_source_dir_type([k.lower() for k in excess_kwargs if k not in options.__dict__])

    def run(self, *args, grid, warmup, **kwargs):
        self.normalize_runner_kwargs(kwargs)
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        kwargs["debug"] = self.debug

        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        if self.binder is None:
            self.create_binder()

        bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)

        key = ''.join(sig_and_spec) + str((constexpr_vals, excess_kwargs))
        kernel = self.cache[device].get(key, None)

        if kernel is None:
            target = driver.active.get_current_target()
            backend = self.make_backend(target)
            options = backend.parse_options(kwargs)

            assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
            assert "device" not in kwargs, "device option is deprecated; current device will be used"
            assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"

            source_dir_type = self.get_source_dir_type(excess_kwargs, options)

            bound_vals = tuple(bound_args.values())

            sigkeys = [self.params[i].name for i in self.non_constexpr_indices]
            sigvals = sig_and_spec[:len(sigkeys)]
            signature = {k: ('*i8' if (v == 'none') else v) for (k, v) in zip(sigkeys, sigvals)}

            configs = (self._get_config(*bound_vals), )
            constants = {
                p.name: v
                for (v, p) in zip(bound_vals, self.params)
                if p.is_constexpr or p.num in configs[0].equal_to_1 or v is None
            }
            for i, arg in constants.items():
                if callable(arg):
                    raise TypeError(f"Callable constexpr at index {i} is not supported")

            if self._call_hook(key, signature, device, constants, options, configs):
                return None
            ast_src = self.ASTSource(self, signature, constants, configs[0])
            src, metadata_json = self.get_src_and_metadata_json(kwargs, source_dir_type, None, ast_src)
            kernel = native_compile(src, ast_src, metadata_json, target=target, options=options.__dict__, source_path=self.source_path)
            self.cache[device][key] = kernel

        # Check that used global values have not changed.
        not_present = object()
        for (name, globals_dict_id), (val, globals_dict) in self.used_global_vals.items():
            if (newVal := globals_dict.get(name, not_present)) != val:
                raise RuntimeError(
                    f"Global variable {name} has changed since we compiled this kernel, from {val} to {newVal}")

        if not warmup:
            assert grid is not None
            grid, grid_0, grid_1, grid_2 = self._resolve_grid(grid, bound_args)
            launch_metadata = kernel.launch_metadata(grid, stream, *non_constexpr_vals)
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       self.CompiledKernel.launch_enter_hook, self.CompiledKernel.launch_exit_hook, *non_constexpr_vals)
        return kernel
