"""Blackwell (sm120, e.g. RTX 5090) warp-specialized attention in TLX.

Two-pass flash attention with a producer task (TMA loads of K/V) and a
consumer task (softmax + tl.dot MMAs on register tiles). The warp-spec,
TMA descriptor loads, shared-memory staging and mbarrier synchronization are
all TLX; the MMA uses tl.dot because fbtriton 3.7.4's TMEM (tcgen05) load
path does not lower on sm120 (`tcgen05.wait.ld` is not selectable).

Run with `pip install fbtriton==3.7.4` on an sm120 GPU:

    python examples/runner/python/tlx/blackwell-sm120-fa-ws.py
"""
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor
import triton_runner

triton_runner.configure_jit_backend()

DEVICE = triton_runner.torch_utils.get_active_torch_device()


@triton.jit
def _fa_ws_sm120(sm_scale, M, desc_q, desc_k, desc_v, desc_o, ZH, N_CTX,
                 HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # shared-memory staging: q stays resident, k/v double-buffered
    q_tile = tlx.local_alloc((BLOCK_M, HEAD_DIM), tlx.dtype_of(desc_q), 1)
    k_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), 2)
    v_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_v), 2)

    q_full = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    k_fulls = tlx.alloc_barriers(num_barriers=2, arrive_count=1)
    v_fulls = tlx.alloc_barriers(num_barriers=2, arrive_count=1)
    k_empties = tlx.alloc_barriers(num_barriers=2, arrive_count=1)
    v_empties = tlx.alloc_barriers(num_barriers=2, arrive_count=1)

    with tlx.async_tasks():
        # producer: TMA-load q once, then stream double-buffered k/v
        with tlx.async_task("default"):
            start_m = tl.program_id(0)
            off_hz = tl.program_id(1)
            offset_y = off_hz * N_CTX
            tlx.barrier_expect_bytes(q_full[0], 2 * BLOCK_M * HEAD_DIM)
            tlx.async_descriptor_load(desc_q, q_tile[0], [offset_y + start_m * BLOCK_M, 0], q_full[0])
            acc_cnt = 0
            k_phase = 0
            v_phase = 0
            for _ in tl.range(0, N_CTX, BLOCK_N):
                buf = acc_cnt % 2
                k_phase = k_phase ^ (buf == 0)
                v_phase = v_phase ^ (buf == 0)
                tlx.barrier_wait(k_empties[buf], k_phase)
                tlx.barrier_expect_bytes(k_fulls[buf], 2 * BLOCK_N * HEAD_DIM)
                tlx.async_descriptor_load(desc_k, k_tiles[buf], [offset_y + acc_cnt * BLOCK_N, 0], k_fulls[buf])
                tlx.barrier_wait(v_empties[buf], v_phase)
                tlx.barrier_expect_bytes(v_fulls[buf], 2 * BLOCK_N * HEAD_DIM)
                tlx.async_descriptor_load(desc_v, v_tiles[buf], [offset_y + acc_cnt * BLOCK_N, 0], v_fulls[buf])
                acc_cnt += 1

        # consumer: online-softmax attention, tl.dot MMAs on register tiles
        with tlx.async_task(num_warps=4):
            qk_scale = sm_scale * 1.44269504
            start_m = tl.program_id(0)
            off_hz = tl.program_id(1)
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

            tlx.barrier_wait(q_full[0], 0)
            q = tlx.local_load(q_tile[0])

            m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            acc_cnt = 0
            for _ in tl.range(0, N_CTX, BLOCK_N):
                buf = acc_cnt % 2
                phase = acc_cnt // 2 % 2
                tlx.barrier_wait(k_fulls[buf], phase)
                k = tlx.local_load(k_tiles[buf])
                k_t = tl.trans(k)
                qk = tl.dot(q, k_t, out_dtype=tl.float32) * qk_scale
                m_new = tl.maximum(m_i, tl.max(qk, 1))
                alpha = tl.math.exp2(m_i - m_new)
                p = tl.math.exp2(qk - m_new[:, None])
                l_i = l_i * alpha + tl.sum(p, 1)
                tlx.barrier_arrive(k_empties[buf], 1)
                tlx.barrier_wait(v_fulls[buf], phase)
                v = tlx.local_load(v_tiles[buf])
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
                tlx.barrier_arrive(v_empties[buf], 1)
                m_i = m_new
                acc_cnt += 1

            # epilogue
            acc = acc / l_i[:, None]
            tl.store(M + off_hz * N_CTX + offs_m, m_i + tl.math.log2(l_i))
            desc_o.store([off_hz * N_CTX + start_m * BLOCK_M, 0], acc.to(tlx.dtype_of(desc_o)))


def attention(q, k, v, sm_scale):
    Z, H, N_CTX, HEAD_DIM = q.shape
    o = torch.empty_like(q)
    M = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)
    y_dim = Z * H * N_CTX

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    dummy = [1, 1]
    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy)
    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy)
    desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy)
    desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy)

    BLOCK_M, BLOCK_N = 64, 64
    desc_q.block_shape = [BLOCK_M, HEAD_DIM]
    desc_k.block_shape = [BLOCK_N, HEAD_DIM]
    desc_v.block_shape = [BLOCK_N, HEAD_DIM]
    desc_o.block_shape = [BLOCK_M, HEAD_DIM]

    def grid(META):
        return (triton.cdiv(N_CTX, BLOCK_M), Z * H, 1)

    _fa_ws_sm120[grid](
        sm_scale, M,
        desc_q, desc_k, desc_v, desc_o,
        Z * H, N_CTX,
        HEAD_DIM=HEAD_DIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=1,
        cubin_dir=triton_runner.get_file_dir(__file__),
    )
    return o


if __name__ == "__main__":
    torch.manual_seed(20)
    Z, H, N_CTX, HEAD_DIM = 1, 4, 1024, 64
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE).normal_(0, 0.5)
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE).normal_(0, 0.5)
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE).normal_(0, 0.5)
    sm_scale = 1.0 / HEAD_DIM**0.5

    o = attention(q, k, v, sm_scale)
    torch.cuda.synchronize()

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale)
    diff = (o.float() - ref.float()).abs().max().item()
    print(f"dispatched class: {type(_fa_ws_sm120).__name__}")
    print(f"max diff vs SDPA: {diff}")
    assert diff < 0.05, f"MISMATCH: {diff}"
    print("✅ Blackwell sm120 TLX warp-spec attention matches SDPA")
