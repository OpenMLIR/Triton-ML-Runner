from typing import Callable, Dict, Iterable, Optional, Union, overload

from triton.runtime.jit import JITFunction, KernelInterface, T
from triton.runtime.jit import compute_cache_key
from triton.runtime import driver
from triton import knobs
from collections import defaultdict
from pathlib import Path
from .. import TRITON_RUNNER_PROD_TEST
from ..tvm_ffi import CompiledTVMFFIKernel

_kernel_cache_dirs: Dict[str, set] = defaultdict(set)


def track_kernel_cache_dir(kernel, name):
    from ..debug.console import blue_print, red_print
    from triton.runtime.cache import get_cache_manager
    cache_dir = get_cache_manager(kernel.hash).cache_dir
    old_num = len(_kernel_cache_dirs[name])
    _kernel_cache_dirs[name].add(cache_dir)
    if old_num != len(_kernel_cache_dirs[name]):
        blue_print(f"[ProdJIT] {name} cache dir: {cache_dir}")
        if old_num > 0:
            red_print(f"[ProdJIT] {name} has multiple cache dirs: {_kernel_cache_dirs[name]}")


def update_kernel_metadata(kernel, bound_args, specialization):
    import glob
    import json
    import os
    from triton.runtime.cache import get_cache_manager
    from ..compat.version import triton_version
    from .. import __version__
    kernel_signature = tuple((k, arg_type, spec) for k, (arg_type, spec) in zip(bound_args.keys(), specialization))
    kernel_cache_dir = get_cache_manager(kernel.hash).cache_dir
    json_files = [f for f in glob.glob(os.path.join(kernel_cache_dir, "*.json")) if not os.path.basename(f).startswith("__grp__")]
    if json_files:
        json_path = Path(json_files[0])
        meta = json.loads(json_path.read_text())
        runner_meta = {
            "kernel_signature": str(kernel_signature),
            "triton_version": triton_version,
            "triton_runner_version": __version__,
            **meta,
        }
        json_path.write_text(json.dumps(runner_meta))
    return runner_meta

class ProdJITFunction(JITFunction[KernelInterface[T]]):

    def run(self, *args, grid, warmup, **kwargs):
        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
        kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode

        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
        # specialization is list[tuple[str, Any]], where first element of tuple is
        # the type and the second parameter is the 'specialization' value.
        bound_args, specialization, options = binder(*args, **kwargs)

        # add a cache field to the kernel specializations for kernel specific
        # pass pipelines
        if knobs.runtime.add_stages_inspection_hook is not None:
            inspect_stages_key, inspect_stages_hash = knobs.runtime.add_stages_inspection_hook()
            specialization.append(f'("custom_pipeline", {inspect_stages_hash})')

        key = compute_cache_key(kernel_key_cache, specialization, options)
        tvm_key = key + "_tvm"
        kernel = kernel_cache.get(key, None)

        # Kernel is not cached; we have to compile.
        if kernel is None:
            options, signature, constexprs, attrs = self._pack_args(backend, kwargs, bound_args, specialization,
                                                                    options)
            object.__setattr__(options, "tvm", True)

            kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)
            if kernel is None:
                return None

        if hasattr(kernel, "result"):
            kernel = kernel.result()
            kernel_cache[key] = kernel

        tvm_kernel = kernel_cache.get(tvm_key, None)
        if tvm_kernel is None:
            kernel._init_handles()
            runner_metadata = update_kernel_metadata(kernel, bound_args, specialization)
            tvm_kernel = CompiledTVMFFIKernel(kernel.function, runner_metadata)
            tvm_kernel._get_launcher()
            kernel_cache[tvm_key] = tvm_kernel

        if TRITON_RUNNER_PROD_TEST:
            track_kernel_cache_dir(kernel, self.__name__)

        # Check that used global values have not changed.
        not_present = object()
        for (name, _), (val, globals_dict) in self.used_global_vals.items():
            if (newVal := globals_dict.get(name, not_present)) != val:
                raise RuntimeError(
                    f"Global variable {name} has changed since we compiled this kernel, from {val} to {newVal}")

        if not warmup:
            # canonicalize grid
            assert grid is not None
            if callable(grid):
                grid = grid(bound_args)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1
            # launch kernel via TVM-FFI
            tvm_kernel.run(grid_0, grid_1, grid_2,
                           knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook,
                           *bound_args.values())
        return kernel


# -----------------------------------------------------------------------------
# `jit` decorator
# -----------------------------------------------------------------------------


@overload
def jit(fn: T) -> JITFunction[T]:
    ...


@overload
def jit(
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[T], ProdJITFunction[T]]:
    ...


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[ProdJITFunction[T], Callable[[T], ProdJITFunction[T]]]:
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, arguments are
        implicitly converted to pointers if they have a :code:`.data_ptr()` method
        and a `.dtype` attribute.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * builtins within the triton package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """

    def decorator(fn: T) -> ProdJITFunction[T]:
        assert callable(fn)
        if knobs.runtime.interpret:
            from ..interpreter import InterpretedFunction
            return InterpretedFunction(fn, version=version, do_not_specialize=do_not_specialize,
                                       do_not_specialize_on_alignment=do_not_specialize_on_alignment, debug=debug,
                                       noinline=noinline, repr=repr, launch_metadata=launch_metadata)
        else:
            return ProdJITFunction(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )

    if fn is not None:
        return decorator(fn)

    else:
        return decorator
