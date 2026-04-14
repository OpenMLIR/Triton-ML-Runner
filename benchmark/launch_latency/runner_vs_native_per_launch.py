import argparse
import dataclasses
import os
from collections.abc import Callable

os.environ.setdefault("TRITON_RUNNER_PROD", "1")

import torch
import triton

import triton_runner
from triton_runner.bench.launch_latency.kernels import (
    nop_kernel,
    nop_with_args_kernel,
    runner_nop_kernel,
    runner_nop_with_args_kernel,
)
from triton_runner.bench.matmul.arch import (
    ArchConfig,
    DOT_BLOCKS,
    SCALAR_BLOCKS,
    TMA_BLOCKS,
    enable_tma_allocator,
    reference_matmul,
    resolve_arch_config,
)
from triton_runner.bench.matmul.kernels import (
    _dot_matmul_kernel,
    _scalar_matmul_kernel,
    _tma_matmul_kernel,
)
from triton_runner.bench.utils import bench_host_us


plain_scalar_matmul_kernel = triton.jit(_scalar_matmul_kernel)
plain_dot_matmul_kernel = triton.jit(_dot_matmul_kernel)
plain_tma_matmul_kernel = triton.jit(_tma_matmul_kernel)

triton_runner.configure_jit_backend()

runner_scalar_matmul_kernel = triton.jit(_scalar_matmul_kernel)
runner_dot_matmul_kernel = triton.jit(_dot_matmul_kernel)
runner_tma_matmul_kernel = triton.jit(_tma_matmul_kernel)


@dataclasses.dataclass(frozen=True)
class LaunchCase:
    name: str
    native_launch: Callable[[], None]
    runner_launch: Callable[[], None]


def _make_nop_case(with_args: bool) -> LaunchCase:
    if not with_args:
        return LaunchCase(
            name="nop",
            native_launch=lambda: nop_kernel[(1,)](),
            runner_launch=lambda: runner_nop_kernel[(1,)](),
        )

    tensors = [torch.zeros(1, device="cuda") for _ in range(5)]
    ints = [1 for _ in range(9)]
    constexprs = [32 for _ in range(5)]
    call_args = tuple([*tensors, *ints, *constexprs])
    return LaunchCase(
        name="nop-with-args",
        native_launch=lambda: nop_with_args_kernel[(1,)](*call_args),
        runner_launch=lambda: runner_nop_with_args_kernel[(1,)](*call_args),
    )


def _make_scalar_matmul_case(
    a: torch.Tensor,
    b: torch.Tensor,
    c_native: torch.Tensor,
    c_runner: torch.Tensor,
    m: int,
    n: int,
    k: int,
) -> LaunchCase:
    block_m, block_n = SCALAR_BLOCKS
    grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m))
    native_args = (
        a,
        b,
        c_native,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c_native.stride(0),
        c_native.stride(1),
    )
    runner_args = (
        a,
        b,
        c_runner,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c_runner.stride(0),
        c_runner.stride(1),
    )
    meta = {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n}
    return LaunchCase(
        name="python-matmul-scalar",
        native_launch=lambda: plain_scalar_matmul_kernel[grid](*native_args, **meta),
        runner_launch=lambda: runner_scalar_matmul_kernel[grid](*runner_args, **meta),
    )


def _make_dot_matmul_case(
    a: torch.Tensor,
    b: torch.Tensor,
    c_native: torch.Tensor,
    c_runner: torch.Tensor,
    m: int,
    n: int,
    k: int,
) -> LaunchCase:
    block_m, block_k, block_n = DOT_BLOCKS
    grid = (triton.cdiv(n, block_k), triton.cdiv(m, block_m))
    native_args = (
        a,
        b,
        c_native,
        m,
        k,
        n,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c_native.stride(0),
        c_native.stride(1),
    )
    runner_args = (
        a,
        b,
        c_runner,
        m,
        k,
        n,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c_runner.stride(0),
        c_runner.stride(1),
    )
    meta = {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_K": block_k, "BLOCK_SIZE_N": block_n}
    return LaunchCase(
        name="python-matmul-dot",
        native_launch=lambda: plain_dot_matmul_kernel[grid](*native_args, **meta),
        runner_launch=lambda: runner_dot_matmul_kernel[grid](*runner_args, **meta),
    )


def _make_tma_matmul_case(
    a: torch.Tensor,
    b: torch.Tensor,
    c_native: torch.Tensor,
    c_runner: torch.Tensor,
    m: int,
    n: int,
    k: int,
) -> LaunchCase:
    block_m, block_n, block_k = TMA_BLOCKS
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_k))
    native_args = (a, b, c_native, m, k, n)
    runner_args = (a, b, c_runner, m, k, n)
    meta = {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k}
    return LaunchCase(
        name="python-matmul-tma",
        native_launch=lambda: plain_tma_matmul_kernel[grid](*native_args, **meta),
        runner_launch=lambda: runner_tma_matmul_kernel[grid](*runner_args, **meta),
    )


def _make_matmul_case(m: int, n: int, k: int) -> tuple[LaunchCase, ArchConfig]:
    _, config = resolve_arch_config()
    if config.variant == "tma":
        enable_tma_allocator()

    device = torch.device("cuda")
    a = torch.randn((m, k), device=device, dtype=torch.float32).to(config.input_dtype)
    b = torch.randn((k, n), device=device, dtype=torch.float32).to(config.input_dtype)
    c_native = torch.empty((m, n), device=device, dtype=config.output_dtype)
    c_runner = torch.empty_like(c_native)

    if config.variant == "scalar":
        case = _make_scalar_matmul_case(a, b, c_native, c_runner, m, n, k)
    elif config.variant == "dot":
        case = _make_dot_matmul_case(a, b, c_native, c_runner, m, n, k)
    else:
        case = _make_tma_matmul_case(a, b, c_native, c_runner, m, n, k)

    case.native_launch()
    case.runner_launch()
    torch.cuda.synchronize()
    ref = reference_matmul(a, b, config).float()
    if not torch.allclose(c_native.float(), ref, atol=config.atol, rtol=0):
        raise AssertionError("Native Triton output mismatch.")
    if not torch.allclose(c_runner.float(), ref, atol=config.atol, rtol=0):
        raise AssertionError("triton_runner output mismatch.")
    return case, config


def _measure(case: LaunchCase, warmup: int, iters: int, repeats: int) -> tuple[float, float]:
    case.native_launch()
    case.runner_launch()
    torch.cuda.synchronize()
    native_us = bench_host_us(case.native_launch, warmup=warmup, iters=iters, repeats=repeats)
    runner_us = bench_host_us(case.runner_launch, warmup=warmup, iters=iters, repeats=repeats)
    return native_us, runner_us


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the average extra host launch time per kernel launch for triton_runner vs native Triton."
    )
    parser.add_argument("--scenario", choices=("nop", "nop-with-args", "matmul"), default="nop-with-args")
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

    config = None
    if args.scenario == "nop":
        case = _make_nop_case(with_args=False)
    elif args.scenario == "nop-with-args":
        case = _make_nop_case(with_args=True)
    else:
        case, config = _make_matmul_case(args.m, args.n, args.k)

    native_us, runner_us = _measure(case, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    delta_us = runner_us - native_us
    delta_ns = delta_us * 1e3
    delta_pct = (delta_us / native_us * 100.0) if native_us else 0.0

    print(f"device: {props.name} (cc {props.major}.{props.minor})")
    print(f"scenario: {case.name}")
    if config is not None:
        print(f"problem: A=({args.m}, {args.k}) B=({args.k}, {args.n}) C=({args.m}, {args.n}) variant={config.variant}")
    print(f"runner env: prod={os.environ.get('TRITON_RUNNER_PROD', '0')} prod_test={os.environ.get('TRITON_RUNNER_PROD_TEST', '0')}")
    print(f"measure: host launch latency via kernel[grid](), median over {args.repeats} repeats")
    print(f"native Triton: {native_us:.3f} us/launch")
    print(f"triton_runner: {runner_us:.3f} us/launch")
    print(f"extra per launch: {delta_us:.3f} us ({delta_ns:.1f} ns)")
    print(f"runner slowdown: {delta_pct:.2f}%")


if __name__ == "__main__":
    main()
