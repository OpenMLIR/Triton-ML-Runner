"""FlashAttention w/support for learned sinks and banded attention.

This is an expanded version of the Flash Attention v2 implementation (see https://tridao.me/publications/flash2/flash2.pdf)
which can be found at https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html.

This version has been extended to support banded attention and learned attention sinks.

Benchmark: host launch latency — Triton vs TVM-Triton vs direct launch.
"""

import argparse
import dataclasses
import os
from collections.abc import Callable

import pytest
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

import triton_runner
from triton_runner.bench.matmul.arch import enable_tma_allocator
from triton_runner.bench.utils import bench_host_us, make_compiled_launch, make_direct_launch


def attention_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float = 0.125,
    sliding_window: int | None = None,
    start_q: torch.LongTensor = 0,
):
    batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim = query.shape
    batch_size, num_keys, num_key_value_heads, head_dim = key.shape

    sinks = sinks.view(1, num_key_value_heads, num_key_value_groups, 1, 1).float()
    key = key.unsqueeze(3)
    value = value.unsqueeze(3)

    pos_keys = torch.arange(num_keys, device=query.device)
    pos_queries = torch.arange(num_queries, device=query.device) + start_q
    mask = pos_keys[None, :] > pos_queries[:, None]
    mask = mask.float().masked_fill(mask, float("-inf"))

    if sliding_window:
        too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
        mask.masked_fill_(too_old, float("-inf"))

    logits = torch.einsum("bqhmd,bkhmd->bhmqk", query.float(), key.float()) * sm_scale
    logits = logits + mask[None, None, None, :, :]

    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    logits_or_sinks_max = torch.maximum(sinks, logits_max)
    sinks = torch.exp(sinks - logits_or_sinks_max)
    unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
    normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks
    scores = unnormalized_scores / normalizer

    output = torch.einsum("bhmqk,bkhmd->bqhmd", scores, value.float())

    output = output.reshape(batch_size, num_queries, num_key_value_heads * num_key_value_groups * head_dim).bfloat16()
    return output


def _attn_fwd_kernel(
    Q,
    K,
    V,
    Sinks,
    sm_scale,
    M,
    Out,  #
    Start_q,
    Z,
    H,
    N_Q_CTX,
    N_KV_CTX,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BANDWIDTH: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_q = tl.load(Start_q).to(tl.int32)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # load attention sinks
    if Sinks is not None:
        sink = tl.load(Sinks + off_h).to(tl.float32)
    else:
        sink = 0

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + sink
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    q = Q.load([off_z, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])

    if BANDWIDTH:
        lo, hi = tl.maximum(0, start_q + start_m * BLOCK_M - BANDWIDTH), start_q + (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, start_q + (start_m + 1) * BLOCK_M
    hi = tl.minimum(hi, N_KV_CTX)

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        mask = (start_n + offs_n)[None, :] > (start_q + offs_m)[:, None]

        if BANDWIDTH:
            too_old = (start_n + offs_n[None, :]) < (start_q + offs_m[:, None] - BANDWIDTH + 1)
            mask = mask | too_old

        k = K.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM]).T
        qk = tl.dot(q, k, allow_tf32=False)

        qk = qk * qk_scale + tl.where(mask, -1.0e6, 0.0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp(qk)
        alpha = tl.math.exp(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = V.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM])
        v = v.to(tl.float32)
        acc = tl.dot(p, v, acc, allow_tf32=False)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    sink = tl.math.exp(sink - m_i)
    z = l_i + sink
    acc = acc / z[:, None]
    # m_i += tl.math.log(l_i)
    # m_ptrs = M + off_hz * N_Q_CTX + offs_m
    # tl.store(m_ptrs, m_i)
    acc = acc.to(Out.dtype)[None, None, :, :]
    Out.store([off_z, off_h, start_m * BLOCK_M, 0], acc)


plain_attn_fwd = triton.jit(_attn_fwd_kernel)
_attn_fwd = plain_attn_fwd  # used by _attention.forward

triton_runner.configure_jit_backend()

runner_attn_fwd = triton.jit(_attn_fwd_kernel)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, sm_scale, bandwidth, start_q):
        assert len(start_q) == 1
        bs, n_ctx, n_kv_heads, repeat_kv, HEAD_DIM_Q = q.shape
        bs, n_kv_ctx, n_kv_heads, HEAD_DIM_K = k.shape
        bs, n_kv_ctx, n_kv_heads, HEAD_DIM_V = v.shape
        n_heads = n_kv_heads * repeat_kv
        q = q.view(bs, n_ctx, n_heads, HEAD_DIM_Q)
        k = k.view(bs, n_kv_ctx, n_kv_heads, HEAD_DIM_K)
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        q = q.transpose(1, 2).contiguous()
        k = k.repeat_interleave(repeat_kv, dim=2).transpose(1, 2).contiguous()
        v = v.repeat_interleave(repeat_kv, dim=2).transpose(1, 2).contiguous()

        BLOCK_M = 64
        BLOCK_N = 64
        m_pad_size = BLOCK_M - n_ctx % BLOCK_M if n_ctx % BLOCK_M != 0 else 0
        # pad q to multiple of its block size in the n_ctx dimension (-2)
        q = torch.nn.functional.pad(q, (0, 0, 0, m_pad_size))
        n_pad_size = BLOCK_N - n_kv_ctx % BLOCK_N if n_kv_ctx % BLOCK_N != 0 else 0
        # pad k and v to multiple of their block size in the n_kv_ctx dimension
        k = torch.nn.functional.pad(k, (0, 0, 0, n_pad_size))
        v = torch.nn.functional.pad(v, (0, 0, 0, n_pad_size))

        o = torch.empty_like(q)
        M = torch.empty((bs, n_heads, n_ctx + m_pad_size), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(n_ctx, BLOCK_M), bs * n_heads, 1)
        _attn_fwd[grid](
            TensorDescriptor.from_tensor(q, [1, 1, BLOCK_M, HEAD_DIM_K]),
            TensorDescriptor.from_tensor(k, [1, 1, BLOCK_N, HEAD_DIM_K]),
            TensorDescriptor.from_tensor(v, [1, 1, BLOCK_N, HEAD_DIM_K]),
            sinks,
            sm_scale,
            M,
            TensorDescriptor.from_tensor(o, [1, 1, BLOCK_M, HEAD_DIM_K]),
            start_q,
            q.shape[0],
            q.shape[1],
            N_Q_CTX=n_ctx + m_pad_size,
            N_KV_CTX=n_kv_ctx,
            HEAD_DIM=HEAD_DIM_K,
            BANDWIDTH=bandwidth,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=2,
        )

        # ctx.save_for_backward(q, k, v, sinks, o, M, start_q)
        # ctx.sm_scale = sm_scale
        # ctx.bandwidth = bandwidth

        o = o[:, :, :n_ctx, :].transpose(1, 2).contiguous()
        o = o.view(bs, n_ctx, n_heads * HEAD_DIM_V)
        return o


attention = _attention.apply


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_queries", [1, 16])
@pytest.mark.parametrize("num_keys", [128, 32])
@pytest.mark.parametrize("num_key_value_heads", [8])
@pytest.mark.parametrize("num_key_value_groups", [8])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("sm_scale", [0.125])
@pytest.mark.parametrize("sliding_window", [None, 128])
@pytest.mark.parametrize("start_q", [0, 5])
def test_eq(batch_size, num_queries, num_keys, num_key_value_heads, num_key_value_groups, head_dim, sm_scale, sliding_window, start_q):
    if num_queries > num_keys:
        pytest.skip("too many queries")

    q = torch.randn(batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim).bfloat16().cuda()
    k = torch.randn(batch_size, num_keys, num_key_value_heads, head_dim).bfloat16().cuda()
    v = torch.randn(batch_size, num_keys, num_key_value_heads, head_dim).bfloat16().cuda()
    sinks = torch.randn(num_key_value_heads * num_key_value_groups).bfloat16().cuda()

    start_q = torch.tensor([start_q], dtype=torch.int32).cuda()

    o1 = attention(q, k, v, sinks, sm_scale, sliding_window, start_q)
    o2 = attention_ref(q, k, v, sinks, sm_scale, sliding_window, start_q)

    torch.testing.assert_close(o1, o2)


BLOCK_M = 64
BLOCK_N = 64


@dataclasses.dataclass(frozen=True)
class LaunchPlan:
    grid: tuple[int, int, int]
    compiled_bound_args: tuple[object, ...]
    direct_bound_args: tuple[object, ...]
    plain_launch: Callable[[], None]
    runner_warmup: Callable[[], object]


def _make_launch_plan(bs: int, n_heads: int, n_ctx: int, n_kv_ctx: int, head_dim: int) -> LaunchPlan:
    sm_scale = head_dim ** -0.5
    m_pad = (BLOCK_M - n_ctx % BLOCK_M) % BLOCK_M
    n_pad = (BLOCK_N - n_kv_ctx % BLOCK_N) % BLOCK_N
    n_ctx_pad = n_ctx + m_pad
    n_kv_ctx_pad = n_kv_ctx + n_pad

    device = torch.device("cuda")
    q = torch.randn((bs, n_heads, n_ctx_pad, head_dim), device=device, dtype=torch.bfloat16)
    k = torch.randn((bs, n_heads, n_kv_ctx_pad, head_dim), device=device, dtype=torch.bfloat16)
    v = torch.randn((bs, n_heads, n_kv_ctx_pad, head_dim), device=device, dtype=torch.bfloat16)
    sinks = torch.randn(n_heads, device=device, dtype=torch.bfloat16)
    m_buf = torch.empty((bs, n_heads, n_ctx_pad), device=device, dtype=torch.float32)
    o_plain = torch.empty_like(q)
    o_runner = torch.empty_like(q)
    start_q = torch.tensor([0], dtype=torch.int32, device=device)

    q_desc_plain = TensorDescriptor.from_tensor(q, [1, 1, BLOCK_M, head_dim])
    k_desc = TensorDescriptor.from_tensor(k, [1, 1, BLOCK_N, head_dim])
    v_desc = TensorDescriptor.from_tensor(v, [1, 1, BLOCK_N, head_dim])
    o_desc_plain = TensorDescriptor.from_tensor(o_plain, [1, 1, BLOCK_M, head_dim])

    q_desc_runner = TensorDescriptor.from_tensor(q, [1, 1, BLOCK_M, head_dim])
    o_desc_runner = TensorDescriptor.from_tensor(o_runner, [1, 1, BLOCK_M, head_dim])

    grid = (triton.cdiv(n_ctx, BLOCK_M), bs * n_heads, 1)

    # non-constexpr args: Q, K, V, Sinks, sm_scale, M, Out, Start_q, Z, H, N_Q_CTX, N_KV_CTX
    # constexpr args: HEAD_DIM, BLOCK_M, BLOCK_N, BANDWIDTH
    call_args_plain = (q_desc_plain, k_desc, v_desc, sinks, sm_scale, m_buf, o_desc_plain,
                       start_q, bs, n_heads, n_ctx_pad, n_kv_ctx)
    call_args_runner = (q_desc_runner, k_desc, v_desc, sinks, sm_scale, m_buf, o_desc_runner,
                        start_q, bs, n_heads, n_ctx_pad, n_kv_ctx)
    meta = dict(HEAD_DIM=head_dim, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BANDWIDTH=0, num_stages=2)

    # bound_args = non-constexpr + constexpr (positional), no num_stages (compile-time only)
    compiled_bound_args = (*call_args_runner, head_dim, BLOCK_M, BLOCK_N, 0)
    direct_bound_args = compiled_bound_args

    def plain_launch() -> None:
        plain_attn_fwd[grid](*call_args_plain, **meta)

    def runner_warmup():
        return runner_attn_fwd.warmup(*call_args_runner, grid=grid, **meta)

    return LaunchPlan(grid, compiled_bound_args, direct_bound_args, plain_launch, runner_warmup)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce host launch latency for plain Triton vs TVM-Triton vs direct launch on TMA attention."
    )
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--n_ctx", type=int, default=64)
    parser.add_argument("--n_kv_ctx", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor
    if sm < 90:
        raise RuntimeError(f"TMA attention requires sm90+, got sm{sm}.")

    enable_tma_allocator()

    plan = _make_launch_plan(args.bs, args.n_heads, args.n_ctx, args.n_kv_ctx, args.head_dim)
    compiled_runner = plan.runner_warmup()
    if hasattr(compiled_runner, "result"):
        compiled_runner = compiled_runner.result()
    compiled_launch = make_compiled_launch(compiled_runner, plan.grid, plan.compiled_bound_args)
    direct_launch = make_direct_launch(compiled_runner, plan.grid, plan.direct_bound_args)

    plan.plain_launch()
    compiled_launch()
    direct_launch()
    torch.cuda.synchronize()

    triton_us = bench_host_us(plan.plain_launch, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    tvm_us = bench_host_us(compiled_launch, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    direct_us = bench_host_us(direct_launch, warmup=args.warmup, iters=args.iters, repeats=args.repeats)

    grid_x, grid_y, grid_z = plan.grid
    print(f"device: {props.name} (cc {props.major}.{props.minor})")
    print(f"kernel: sm{sm} python attn-tma")
    print(f"problem: bs={args.bs} n_ctx={args.n_ctx} n_kv_ctx={args.n_kv_ctx} n_heads={args.n_heads} head_dim={args.head_dim} grid=({grid_x}, {grid_y}, {grid_z})")
    print(f"measure: host launch latency, median over {args.repeats} repeats")
    print(f"Triton: {triton_us:.3f} us")
    print(f"TVM-Triton (CompiledTVMFFIKernel.__getitem__/run): {tvm_us:.3f} us")
    print(f"direct launch: {direct_us:.3f} us")
    print(f"TVM-Triton - direct launch: {tvm_us - direct_us:.3f} us")
    print(f"Triton - direct launch: {triton_us - direct_us:.3f} us")


if __name__ == "__main__":
    main()
