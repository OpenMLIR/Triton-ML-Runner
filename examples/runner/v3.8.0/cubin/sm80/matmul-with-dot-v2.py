import triton
import triton.language as tl

import triton_runner
triton_runner.configure_jit_backend()

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bn, stride_bk,
    stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid_k = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        max_idx = N - n * BLOCK_SIZE_N
        a = tl.load(a_ptrs + n * BLOCK_SIZE_N * stride_an, mask=offs_n[None, :] < max_idx, other=0.0)
        b = tl.load(b_ptrs + n * BLOCK_SIZE_N * stride_bn, mask=offs_n[:, None] < max_idx, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck
    c_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul(a, b):
    M, N = a.shape
    N, K = b.shape
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1
    c = torch.randn(M, K, device=device)
    grid = lambda META: (triton.cdiv(K, META['BLOCK_SIZE_K']), triton.cdiv(M, META['BLOCK_SIZE_M']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_K=64,
        BLOCK_SIZE_N=64,
        cubin_dir=triton_runner.get_file_dir(__file__)
    )
    return c

M, N, K = 8192, 6144, 4096

import torch
device = torch.cuda.current_device()

A = torch.randn(M, N, device=device, dtype=torch.float16)
B = torch.randn(N, K, device=device, dtype=torch.float16)

triton_output = matmul(A, B)
torch_output = torch.matmul(A, B)

if torch.allclose(triton_output.float(), torch_output.float(), atol=0.125, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")