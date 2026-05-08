from typing import Callable, Dict, Iterable, Optional, Union, overload

from operator import itemgetter

from triton.runtime.jit import JITFunction, KernelInterface, T
from triton.runtime.jit import compute_cache_key
from triton.runtime import driver
from triton import knobs

_DRIVER_ACTIVE = driver.active

# Direct CUDA bindings for the hot path: triton.runtime.driver.active.get_current_device
# wraps torch.cuda.current_device which goes through _lazy_init/is_initialized checks
# every call. Once the process is past first init those checks are pure overhead, so
# the fast path uses the C entries directly.
import torch as _torch  # noqa: E402
_GET_RAW_STREAM = _torch._C._cuda_getCurrentRawStream
_GET_RAW_DEVICE = _torch._C._cuda_getDevice


class _ProdLaunchProxy:
    """Per-launch proxy returned from ProdJITFunction.__getitem__.

    Pre-canonicalizes the grid (when not callable) so the per-launch hot path
    avoids re-doing len()/index checks, and dispatches directly via the
    JITFunction's fast cache without the extra lambda frame.
    """

    __slots__ = ("_jit", "_grid", "_g0", "_g1", "_g2", "_callable", "_warm")

    def __init__(self, jit, grid):
        self._jit = jit
        self._grid = grid
        self._warm = None
        if callable(grid):
            self._callable = True
        else:
            self._callable = False
            n = len(grid)
            self._g0 = grid[0]
            self._g1 = grid[1] if n > 1 else 1
            self._g2 = grid[2] if n > 2 else 1

    def __call__(self, *args, **kwargs):
        # Hottest path: proxy was already warmed up with a matching call signature.
        warm = self._warm
        if warm is not None and not kwargs:
            warm_types, int_getter, warm_int_values, g0, g1, g2, launch_c, kernel = warm
            if tuple(map(type, args)) == warm_types:
                int_values = int_getter(args) if int_getter is not None else ()
                if int_values == warm_int_values:
                    launch_c(g0, g1, g2,
                             _GET_RAW_STREAM(_GET_RAW_DEVICE()),
                             *args, 0, 0)
                    return kernel

        # Resolve via JIT-level dispatch dict, populate warm cache on hit.
        jit = self._jit
        disp = jit._fast_dispatch
        if (disp is not None
                and not kwargs
                and not self._callable
                and knobs.runtime.add_stages_inspection_hook is None
                and not jit.used_global_vals):
            arg_types = tuple(map(type, args))
            sub = disp.get(arg_types)
            if sub is not None:
                int_getter, sub_dict = sub
                int_values = int_getter(args) if int_getter is not None else ()
                slot = sub_dict.get(int_values)
                if slot is not None:
                    launch_c, kernel = slot
                    self._warm = (arg_types, int_getter, int_values,
                                  self._g0, self._g1, self._g2, launch_c, kernel)
                    launch_c(self._g0, self._g1, self._g2,
                             _GET_RAW_STREAM(_GET_RAW_DEVICE()),
                             *args, 0, 0)
                    return kernel
        return jit.run(*args, grid=self._grid, warmup=False, **kwargs)
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

    _fast_dispatch: Dict[tuple, tuple] = None
    _proxy_cache: Dict[object, "_ProdLaunchProxy"] = None

    def __getitem__(self, grid):
        cache = self._proxy_cache
        if cache is None:
            cache = {}
            self._proxy_cache = cache
        try:
            proxy = cache.get(grid)
        except TypeError:
            return _ProdLaunchProxy(self, grid)
        if proxy is None:
            proxy = _ProdLaunchProxy(self, grid)
            cache[grid] = proxy
        return proxy

    def run(self, *args, grid, warmup, **kwargs):
        # Fast path: skip binder + cache_key + globals check on warm cache.
        # Conditions: no warmup, no kwargs, no inspection hook, no globals to verify,
        # and a non-callable grid. Two-tier cache: types -> (int_getter, sub_dict).
        if (not warmup
                and not kwargs
                and self._fast_dispatch is not None
                and knobs.runtime.add_stages_inspection_hook is None
                and not self.used_global_vals
                and not callable(grid)):
            sub = self._fast_dispatch.get(tuple(map(type, args)))
            if sub is not None:
                int_getter, sub_dict = sub
                slot = sub_dict.get(int_getter(args) if int_getter is not None else ())
                if slot is not None:
                    launch_c, kernel = slot
                    drv = _DRIVER_ACTIVE
                    stream = drv.get_current_stream(drv.get_current_device())
                    grid_size = len(grid)
                    launch_c(
                        grid[0],
                        grid[1] if grid_size > 1 else 1,
                        grid[2] if grid_size > 2 else 1,
                        stream, *args, 0, 0,
                    )
                    return kernel

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

        tvm_launcher = getattr(kernel, "_tvm_launcher", None)
        if tvm_launcher is None:
            kernel._init_handles()
            runner_metadata = update_kernel_metadata(kernel, bound_args, specialization)
            tvm_kernel = CompiledTVMFFIKernel(kernel.function, runner_metadata)
            tvm_launcher = tvm_kernel._get_launcher()
            kernel._tvm_launcher = tvm_launcher

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
            # launch kernel via TVM-FFI (fast path inlined)
            fast_eligible = (
                not tvm_launcher._tensordesc_expansion_info
                and tvm_launcher._global_scratch_size == 0
                and tvm_launcher._profile_scratch_size == 0
            )
            if fast_eligible:
                launch_c = tvm_launcher._launch_bound_args_for_tvm_ffi
                launch_c(grid_0, grid_1, grid_2, stream, *bound_args.values(), 0, 0)
                # Populate fast dispatch cache.
                if not kwargs.keys() - {"debug", "instrumentation_mode"} \
                        and knobs.runtime.add_stages_inspection_hook is None \
                        and not self.used_global_vals \
                        and len(args) == len(bound_args):
                    int_t = int
                    arg_types = tuple(map(type, args))
                    int_positions = tuple(i for i, t in enumerate(arg_types) if t is int_t)
                    int_values = tuple(args[i] for i in int_positions)
                    int_getter = itemgetter(*int_positions) if int_positions else None
                    if int_getter is not None and len(int_positions) == 1:
                        # itemgetter(i) returns scalar; wrap to keep tuple keying
                        _single_idx = int_positions[0]
                        int_getter = lambda a, _i=_single_idx: (a[_i],)
                    if self._fast_dispatch is None:
                        self._fast_dispatch = {}
                    sub = self._fast_dispatch.get(arg_types)
                    if sub is None:
                        sub = (int_getter, {})
                        self._fast_dispatch[arg_types] = sub
                    sub[1][int_values] = (launch_c, kernel)
            else:
                tvm_launcher.launch(grid_0, grid_1, grid_2,
                                    *bound_args.values(), stream=stream)
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
