# This file is adapted from:
# https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/utils/triton_op.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import statistics
import time
from collections.abc import Callable

import torch
from triton_runner.testing import do_bench


class TimerContext:

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.elapsed_ms = None

    def __enter__(self):
        if self.enabled:
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs):
        if self.enabled:
            end_time = time.perf_counter()
            self.elapsed_ms = (end_time - self._start_time) * 1e3


def benchmark(name, unit_name="ms"):

    def decorator(method):

        def wrapper(self, *args, **kwargs):
            if kwargs.pop("enable_benchmark", True) is not False:
                input_iter = list(self.get_input_iter())
                # sum_time = 0
                input_len = len(input_iter)
                for idx, input_args in enumerate(input_iter):
                    fn = method(self, *input_args)
                    elapsed_time = do_bench(fn)
                    elapsed_time_str = f"{elapsed_time:8.3f} ms"
                    if unit_name == "us":
                        elapsed_time_str = f"{elapsed_time * 1e3:8.3f} us"
                    if idx == input_len - 1:
                        print(f"[{name:<50}|] time: {elapsed_time_str}")
                    # sum_time += elapsed_time
                # print(f"[{name + " average":<30}|] time: {sum_time/input_len:.6f} ms")
            else:
                return method(self, *args[0])

        return wrapper

    return decorator


def bench_host_us(fn: Callable[[], None], *, warmup: int, iters: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / iters)
    return statistics.median(samples)


def make_compiled_launch(compiled_kernel, grid: tuple, bound_args: tuple[object, ...]) -> Callable[[], None]:
    grid_x = grid[0]
    grid_y = grid[1] if len(grid) > 1 else 1
    grid_z = grid[2] if len(grid) > 2 else 1
    launcher = compiled_kernel[(grid_x, grid_y, grid_z)]

    def launch() -> None:
        launcher(*bound_args)

    return launch


def make_subscript_launch(compiled_kernel, grid: tuple, bound_args: tuple[object, ...]) -> Callable[[], None]:
    grid_x = grid[0]
    grid_y = grid[1] if len(grid) > 1 else 1
    grid_z = grid[2] if len(grid) > 2 else 1
    g = (grid_x, grid_y, grid_z)

    def launch() -> None:
        compiled_kernel[g](*bound_args)

    return launch


def _has_tensor_descriptor(args: tuple) -> bool:
    try:
        from triton.tools.tensor_descriptor import TensorDescriptor
        return any(isinstance(a, TensorDescriptor) for a in args)
    except ImportError:
        return False


def make_direct_launch(compiled_kernel, grid: tuple, bound_args: tuple[object, ...]) -> Callable[[], None]:
    if not hasattr(compiled_kernel, "_get_launcher"):
        return make_compiled_launch(compiled_kernel, grid, bound_args)

    launcher = compiled_kernel._get_launcher()
    grid_x = grid[0]
    grid_y = grid[1] if len(grid) > 1 else 1
    grid_z = grid[2] if len(grid) > 2 else 1

    direct_args = tuple(arg for entry, arg in zip(launcher._signature, bound_args, strict=True) if not entry.is_constexpr)

    def launch() -> None:
        launcher._tvm_func(launcher._registry_handle, grid_x, grid_y, grid_z, 0, *direct_args, 0, 0)

    return launch
