"""Cross-compile performance-7096 kernel for multiple SM architectures.

Generates cubins for sm75 (T4), sm90 (H100), sm120 (5090) using
``runner_sm`` compile-only mode. Run inside each Triton conda env::

    conda run -n triton-v3-1-0 python compile_cross_sm.py
    conda run -n triton-v3-4-0 python compile_cross_sm.py
"""

import json
import os
from pathlib import Path

import torch
import triton
from triton.runtime.jit import MockTensor
import triton_runner

triton_runner.configure_jit_backend()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = Path(HERE) / "results" / "cubins_cross_sm" / f"triton-v{triton.__version__.replace('.', '-')}"

SM_NAMES = {75: "sm75_T4", 90: "sm90_H100", 120: "sm120_5090"}


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: triton.language.constexpr,
    BLOCK_SIZE_N: triton.language.constexpr,
    BLOCK_SIZE_K: triton.language.constexpr,
    GROUP_SIZE_M: triton.language.constexpr,
):
    pid = triton.language.program_id(axis=0)
    num_pid_m = triton.language.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.language.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + triton.language.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + triton.language.arange(0, BLOCK_SIZE_N)) % N
    offs_k = triton.language.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = triton.language.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=triton.language.float32)
    for k in range(0, triton.language.cdiv(K, BLOCK_SIZE_K)):
        a = triton.language.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = triton.language.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = triton.language.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(triton.language.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + triton.language.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + triton.language.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    triton.language.store(c_ptrs, c, mask=c_mask)


def compile_kernel(sm: int):
    M, N, K = 512, 256, 1024
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    return matmul_kernel.warmup(
        MockTensor(torch.float16, [M, K]),
        MockTensor(torch.float16, [K, N]),
        MockTensor(torch.float16, [M, N]),
        M, N, K,
        K, 1,
        N, 1,
        N, 1,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        runner_sm=sm,
        grid=grid,
    )


def main():
    print(f"GPU: {torch.cuda.get_device_name()} (sm{torch.cuda.get_device_capability()})")
    print(f"Triton: {triton.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    for sm in [75, 90, 120]:
        sm_name = SM_NAMES[sm]
        out_dir = OUT / sm_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{sm_name}] compiling ...")
        try:
            compiled = compile_kernel(sm)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        # .cubin
        cubin_path = out_dir / "matmul_kernel.cubin"
        cubin_path.write_bytes(compiled.asm["cubin"])
        # .json
        meta_path = out_dir / "matmul_kernel.json"
        meta_path.write_text(json.dumps(
            compiled.metadata._asdict() if hasattr(compiled.metadata, "_asdict")
            else vars(compiled.metadata),
            default=vars, indent=2))
        # .ptx (for inspection)
        ptx_path = out_dir / "matmul_kernel.ptx"
        ptx_path.write_text(str(compiled.asm["ptx"]))

        print(f"  cubin: {cubin_path.stat().st_size:,} bytes")
        print(f"  ptx:   {ptx_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
