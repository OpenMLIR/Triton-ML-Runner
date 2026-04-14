import argparse
import dataclasses
import os
from collections.abc import Callable

import torch
import triton

import triton_runner
from triton_runner.bench.matmul.arch import (
    ArchConfig, SCALAR_BLOCKS, DOT_BLOCKS, TMA_BLOCKS,
    resolve_arch_config, enable_tma_allocator, reference_matmul, validate_matmul_outputs,
)
from triton_runner.bench.matmul.kernels import _scalar_matmul_kernel, _dot_matmul_kernel, _tma_matmul_kernel
from triton_runner.bench.utils import bench_host_us, make_compiled_launch, make_direct_launch, make_subscript_launch


plain_scalar_matmul_kernel = triton.jit(_scalar_matmul_kernel)
plain_dot_matmul_kernel = triton.jit(_dot_matmul_kernel)
plain_tma_matmul_kernel = triton.jit(_tma_matmul_kernel)

triton_runner.configure_jit_backend()

runner_scalar_matmul_kernel = triton.jit(_scalar_matmul_kernel)
runner_dot_matmul_kernel = triton.jit(_dot_matmul_kernel)
runner_tma_matmul_kernel = triton.jit(_tma_matmul_kernel)


@dataclasses.dataclass(frozen=True)
class LaunchPlan:
    grid: tuple[int, int]
    compiled_bound_args: tuple[object, ...]
    direct_bound_args: tuple[object, ...]
    plain_launch: Callable[[], None]
    runner_warmup: Callable[[], object]


def _make_scalar_plan(a: torch.Tensor, b: torch.Tensor, c_plain: torch.Tensor, c_runner: torch.Tensor,
                      c_direct: torch.Tensor, m: int, n: int, k: int) -> LaunchPlan:
    block_m, block_n = SCALAR_BLOCKS
    grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m))
    call_args_plain = (
        a, b, c_plain, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_plain.stride(0), c_plain.stride(1),
    )
    call_args_runner = (
        a, b, c_runner, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_runner.stride(0), c_runner.stride(1),
    )
    direct_bound_args = (
        a, b, c_direct, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_direct.stride(0), c_direct.stride(1),
        block_m, block_n,
    )
    compiled_bound_args = (
        a, b, c_runner, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_runner.stride(0), c_runner.stride(1),
        block_m, block_n,
    )
    meta = {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n}

    def plain_launch() -> None:
        plain_scalar_matmul_kernel[grid](*call_args_plain, **meta)

    def runner_warmup():
        return runner_scalar_matmul_kernel.warmup(*call_args_runner, grid=grid, **meta)

    return LaunchPlan(grid, compiled_bound_args, direct_bound_args, plain_launch, runner_warmup)


def _make_dot_plan(a: torch.Tensor, b: torch.Tensor, c_plain: torch.Tensor, c_runner: torch.Tensor,
                   c_direct: torch.Tensor, m: int, n: int, k: int) -> LaunchPlan:
    block_m, block_k, block_n = DOT_BLOCKS
    grid = (triton.cdiv(n, block_k), triton.cdiv(m, block_m))
    call_args_plain = (
        a, b, c_plain, m, k, n,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_plain.stride(0), c_plain.stride(1),
    )
    call_args_runner = (
        a, b, c_runner, m, k, n,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_runner.stride(0), c_runner.stride(1),
    )
    direct_bound_args = (
        a, b, c_direct, m, k, n,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_direct.stride(0), c_direct.stride(1),
        block_m, block_k, block_n,
    )
    compiled_bound_args = (
        a, b, c_runner, m, k, n,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_runner.stride(0), c_runner.stride(1),
        block_m, block_k, block_n,
    )
    meta = {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_K": block_k, "BLOCK_SIZE_N": block_n}

    def plain_launch() -> None:
        plain_dot_matmul_kernel[grid](*call_args_plain, **meta)

    def runner_warmup():
        return runner_dot_matmul_kernel.warmup(*call_args_runner, grid=grid, **meta)

    return LaunchPlan(grid, compiled_bound_args, direct_bound_args, plain_launch, runner_warmup)


def _make_tma_plan(a: torch.Tensor, b: torch.Tensor, c_plain: torch.Tensor, c_runner: torch.Tensor,
                   c_direct: torch.Tensor, m: int, n: int, k: int) -> LaunchPlan:
    block_m, block_n, block_k = TMA_BLOCKS
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_k))
    call_args_plain = (a, b, c_plain, m, k, n)
    call_args_runner = (a, b, c_runner, m, k, n)
    direct_bound_args = (a, b, c_direct, m, k, n, block_m, block_n, block_k)
    compiled_bound_args = (a, b, c_runner, m, k, n, block_m, block_n, block_k)
    meta = {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k}

    def plain_launch() -> None:
        plain_tma_matmul_kernel[grid](*call_args_plain, **meta)

    def runner_warmup():
        return runner_tma_matmul_kernel.warmup(*call_args_runner, grid=grid, **meta)

    return LaunchPlan(grid, compiled_bound_args, direct_bound_args, plain_launch, runner_warmup)


def _make_launch_plan(config: ArchConfig, a: torch.Tensor, b: torch.Tensor, c_plain: torch.Tensor,
                      c_runner: torch.Tensor, c_direct: torch.Tensor, m: int, n: int, k: int) -> LaunchPlan:
    if config.variant == "scalar":
        return _make_scalar_plan(a, b, c_plain, c_runner, c_direct, m, n, k)
    if config.variant == "dot":
        return _make_dot_plan(a, b, c_plain, c_runner, c_direct, m, n, k)
    return _make_tma_plan(a, b, c_plain, c_runner, c_direct, m, n, k)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce host launch latency for plain Triton vs TVM-Triton vs direct launch on Python kernels."
    )
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    sm, config = resolve_arch_config()
    if config.variant == "tma":
        enable_tma_allocator()

    a = torch.randn((args.m, args.k), device=device, dtype=torch.float32).to(config.input_dtype)
    b = torch.randn((args.k, args.n), device=device, dtype=torch.float32).to(config.input_dtype)
    c_plain = torch.empty((args.m, args.n), device=device, dtype=config.output_dtype)
    c_runner = torch.empty_like(c_plain)
    c_direct = torch.empty_like(c_plain)

    plan = _make_launch_plan(config, a, b, c_plain, c_runner, c_direct, args.m, args.n, args.k)
    compiled_runner = plan.runner_warmup()
    if hasattr(compiled_runner, "result"):
        compiled_runner = compiled_runner.result()
    compiled_launch = make_compiled_launch(compiled_runner, plan.grid, plan.compiled_bound_args)
    subscript_launch = make_subscript_launch(compiled_runner, plan.grid, plan.compiled_bound_args)
    direct_launch = make_direct_launch(compiled_runner, plan.grid, plan.direct_bound_args)

    plan.plain_launch()
    compiled_launch()
    subscript_launch()
    direct_launch()
    torch.cuda.synchronize()
    validate_matmul_outputs(reference_matmul(a, b, config), config.atol, c_plain, c_runner, c_direct)

    triton_us = bench_host_us(plan.plain_launch, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    tvm_triton_us = bench_host_us(compiled_launch, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    tvm_subscript_us = bench_host_us(subscript_launch, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    direct_us = bench_host_us(direct_launch, warmup=args.warmup, iters=args.iters, repeats=args.repeats)

    grid_x, grid_y = plan.grid
    print(f"device: {props.name} (cc {props.major}.{props.minor})")
    print(f"kernel: sm{sm} python matmul (no cubin_dir)")
    print(f"problem: A=({args.m}, {args.k}) B=({args.k}, {args.n}) C=({args.m}, {args.n}) grid=({grid_x}, {grid_y}, 1)")
    print(f"measure: host launch latency, median over {args.repeats} repeats")
    print(f"Triton: {triton_us:.3f} us")
    print(f"TVM-Triton (CompiledTVMFFIKernel.__getitem__/run): {tvm_triton_us:.3f} us")
    print(f"TVM-Triton (kernel[grid]() each call): {tvm_subscript_us:.3f} us")
    print(f"direct launch: {direct_us:.3f} us")
    print(f"TVM-Triton - direct launch: {tvm_triton_us - direct_us:.3f} us")
    print(f"Triton - direct launch: {triton_us - direct_us:.3f} us")


if __name__ == "__main__":
    main()
