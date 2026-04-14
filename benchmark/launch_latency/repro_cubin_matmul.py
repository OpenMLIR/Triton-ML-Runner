"""Regression note.

Measured on 2026-03-30 with:
    python benchmark/launch_latency/repro_cubin_matmul.py

Environment:
    device: NVIDIA H20-3e (cc 9.0)
    kernel: sm90 cubin matmul
    problem: A=(1, 1) B=(1, 1) C=(1, 1) grid=(1, 1, 1)
    measure: host launch latency, median over 7 repeats

Results:
    Triton: 14.265 us
    TVM-Triton (CompiledTVMFFIKernel.__getitem__/run): 6.140 us
    direct launch: 6.139 us
    TVM-Triton - direct launch: 0.001 us
    Triton - direct launch: 8.126 us

Environment:
    device: Tesla T4 (cc 7.5)
    kernel: sm75 cubin matmul
    problem: A=(1, 1) B=(1, 1) C=(1, 1) grid=(1, 1, 1)
    measure: host launch latency, median over 7 repeats

Results:
    Triton: 20.012 us
    TVM-Triton (CompiledTVMFFIKernel.__getitem__/run): 6.450 us
    direct launch: 5.260 us
    TVM-Triton - direct launch: 1.190 us
    Triton - direct launch: 14.751 us
"""

import argparse
import dataclasses
import os
from collections.abc import Callable
from pathlib import Path

import torch
import triton

from triton_runner.bench.matmul.arch import (
    ArchConfig, SCALAR_BLOCKS, DOT_BLOCKS, TMA_BLOCKS,
    resolve_arch_config, enable_tma_allocator, reference_matmul, validate_matmul_outputs,
)
from triton_runner.bench.matmul.kernels import _scalar_matmul_kernel, _dot_matmul_kernel, _tma_matmul_kernel
from triton_runner.bench.utils import bench_host_us, make_compiled_launch, make_direct_launch, make_subscript_launch
from triton_runner.tvm_ffi import CompiledTVMFFIKernel

scalar_matmul_kernel = triton.jit(_scalar_matmul_kernel)
dot_matmul_kernel = triton.jit(_dot_matmul_kernel)
tma_matmul_kernel = triton.jit(_tma_matmul_kernel)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cubin_dir_for_sm(sm: int) -> Path:
    cubin_dir = _repo_root() / "examples" / "runner" / "v3.4.0" / "cubin" / f"sm{sm}"
    if not cubin_dir.is_dir():
        raise RuntimeError(f"Expected cubin directory {cubin_dir}")
    return cubin_dir


def _only_file(cubin_dir: Path, pattern: str) -> Path:
    matches = sorted(cubin_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {pattern} under {cubin_dir}, found {len(matches)}")
    return matches[0]


def _resolve_arch_config() -> tuple[int, Path, ArchConfig]:
    sm, config = resolve_arch_config()
    return sm, _cubin_dir_for_sm(sm), config


def _make_tvm_ffi_kernel(cubin_dir: Path) -> CompiledTVMFFIKernel:
    cubin_path = _only_file(cubin_dir, "*.cubin")
    return CompiledTVMFFIKernel.from_cubin(cubin_path)


@dataclasses.dataclass(frozen=True)
class LaunchPlan:
    grid: tuple[int, int]
    compiled_bound_args: tuple[object, ...]
    direct_bound_args: tuple[object, ...]
    triton_launch: Callable[[], None]


def _make_scalar_plan(a: torch.Tensor, b: torch.Tensor, c_triton: torch.Tensor, c_tvm: torch.Tensor, c_direct: torch.Tensor,
                      m: int, n: int, k: int) -> LaunchPlan:
    block_m, block_n = SCALAR_BLOCKS
    grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m))
    compiled_bound_args = (
        a, b, c_tvm, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_tvm.stride(0), c_tvm.stride(1),
        block_m, block_n,
    )
    direct_bound_args = (
        a, b, c_direct, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_direct.stride(0), c_direct.stride(1),
        block_m, block_n,
    )

    def triton_launch() -> None:
        scalar_matmul_kernel[grid](
            a, b, c_triton, m, n, k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_triton.stride(0), c_triton.stride(1),
            BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n,
        )

    return LaunchPlan(grid, compiled_bound_args, direct_bound_args, triton_launch)


def _make_dot_plan(a: torch.Tensor, b: torch.Tensor, c_triton: torch.Tensor, c_tvm: torch.Tensor, c_direct: torch.Tensor,
                   m: int, n: int, k: int) -> LaunchPlan:
    block_m, block_k, block_n = DOT_BLOCKS
    grid = (triton.cdiv(n, block_k), triton.cdiv(m, block_m))
    compiled_bound_args = (
        a, b, c_tvm, m, k, n,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_tvm.stride(0), c_tvm.stride(1),
        block_m, block_k, block_n,
    )
    direct_bound_args = (
        a, b, c_direct, m, k, n,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_direct.stride(0), c_direct.stride(1),
        block_m, block_k, block_n,
    )

    def triton_launch() -> None:
        dot_matmul_kernel[grid](
            a, b, c_triton, m, k, n,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_triton.stride(0), c_triton.stride(1),
            BLOCK_SIZE_M=block_m, BLOCK_SIZE_K=block_k, BLOCK_SIZE_N=block_n,
        )

    return LaunchPlan(grid, compiled_bound_args, direct_bound_args, triton_launch)


def _make_tma_plan(a: torch.Tensor, b: torch.Tensor, c_triton: torch.Tensor, c_tvm: torch.Tensor, c_direct: torch.Tensor,
                   m: int, n: int, k: int) -> LaunchPlan:
    block_m, block_n, block_k = TMA_BLOCKS
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_k))
    compiled_bound_args = (a, b, c_tvm, m, k, n, block_m, block_n, block_k)
    direct_bound_args = (a, b, c_direct, m, k, n, block_m, block_n, block_k)

    def triton_launch() -> None:
        tma_matmul_kernel[grid](a, b, c_triton, m, k, n, BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k)

    return LaunchPlan(grid, compiled_bound_args, direct_bound_args, triton_launch)


def _make_launch_plan(config: ArchConfig, a: torch.Tensor, b: torch.Tensor, c_triton: torch.Tensor, c_tvm: torch.Tensor,
                      c_direct: torch.Tensor, m: int, n: int, k: int) -> LaunchPlan:
    if config.variant == "scalar":
        return _make_scalar_plan(a, b, c_triton, c_tvm, c_direct, m, n, k)
    if config.variant == "dot":
        return _make_dot_plan(a, b, c_triton, c_tvm, c_direct, m, n, k)
    return _make_tma_plan(a, b, c_triton, c_tvm, c_direct, m, n, k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce host launch latency for Triton vs TVM-Triton vs direct launch.")
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
    sm, cubin_dir, config = _resolve_arch_config()
    if config.variant == "tma":
        enable_tma_allocator()

    a = torch.randn((args.m, args.k), device=device, dtype=torch.float32).to(config.input_dtype)
    b = torch.randn((args.k, args.n), device=device, dtype=torch.float32).to(config.input_dtype)
    c_triton = torch.empty((args.m, args.n), device=device, dtype=config.output_dtype)
    c_tvm = torch.empty_like(c_triton)
    c_direct = torch.empty_like(c_triton)

    compiled_kernel = _make_tvm_ffi_kernel(cubin_dir)
    plan = _make_launch_plan(config, a, b, c_triton, c_tvm, c_direct, args.m, args.n, args.k)
    tvm_launch = make_compiled_launch(compiled_kernel, plan.grid, plan.compiled_bound_args)
    tvm_subscript_launch = make_subscript_launch(compiled_kernel, plan.grid, plan.compiled_bound_args)
    direct_launch = make_direct_launch(compiled_kernel, plan.grid, plan.direct_bound_args)

    plan.triton_launch()
    tvm_launch()
    tvm_subscript_launch()
    direct_launch()
    torch.cuda.synchronize()
    validate_matmul_outputs(reference_matmul(a, b, config), config.atol, c_triton, c_tvm, c_direct)

    triton_us = bench_host_us(plan.triton_launch, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    tvm_triton_us = bench_host_us(tvm_launch, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    tvm_subscript_us = bench_host_us(tvm_subscript_launch, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    direct_us = bench_host_us(direct_launch, warmup=args.warmup, iters=args.iters, repeats=args.repeats)

    grid_x, grid_y = plan.grid
    print(f"device: {props.name} (cc {props.major}.{props.minor})")
    print(f"kernel: sm{sm} cubin matmul from {cubin_dir}")
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
