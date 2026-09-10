import triton
import triton.language as tl
import torch

import triton_runner

DEVICE = triton_runner.torch_utils.get_active_torch_device()


triton_runner.configure_jit_backend()

@triton.jit
def matmul_kernel_make_tensor_desciptor(a_ptr, b_ptr, c_ptr,  #
                                        M, N, K,  #
                                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                        BLOCK_SIZE_K: tl.constexpr,  #
                                        ):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(tl.cdiv(N, BLOCK_SIZE_N)):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, n * BLOCK_SIZE_N])
        b = b_desc.load([n * BLOCK_SIZE_N, pid_k * BLOCK_SIZE_K])
        accumulator = tl.dot(a, b, acc=accumulator)

    accumulator = accumulator.to(tl.float16)
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_k * BLOCK_SIZE_K], accumulator)


def matmul(a, b):
    M, N = a.shape
    N, K = b.shape

    # Leading dimensions must be multiples of 16-byte strides
    # if M % 4 == 0 and N % 4 == 0 and K % 4 == 0:

    # Allocates output.
    c = torch.empty((M, K), device=a.device, dtype=torch.float16)

    # TMA descriptors require a global memory allocation
    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(K, META['BLOCK_SIZE_K']), )

    matmul_kernel_make_tensor_desciptor[grid](
        a, b, c,
        M, N, K,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_K=64,
        BLOCK_SIZE_N=64,
        cubin_dir=triton_runner.get_file_dir(__file__)
    )

    return c


M, N, K = 1024, 512, 256

device = torch.cuda.current_device()

a = torch.randn((M, N), device=DEVICE, dtype=torch.float16)
b = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
a_fp8 = a.to(torch.float8_e5m2)
b_fp8 = b.to(torch.float8_e5m2)
triton_output = matmul(a_fp8, b_fp8)
torch_output = torch.matmul(a_fp8.to(torch.float16), b_fp8.to(torch.float16))
print(f"triton_output_with_fp8_inputs={triton_output}")
print(f"torch_output_with_fp8_inputs={torch_output}")
if torch.allclose(triton_output.float(), torch_output.float(), atol=0.125, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
