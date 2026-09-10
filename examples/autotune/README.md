摘要：Triton Runner是针对OpenAI/Triton的多层级Runner工具，提供了IR/PTX/cubin的多层级执行。本文是多层级autotune的使用文档。

项目地址：[ToyAIX/triton-runner](https://github.com/toyaix/triton-runner)。

这个功能以及示例目前在 main分支，还未发布，请使用源码安装。

```shell
git clone https://github.com/toyaix/triton-runner
cd triton-runner

pip install -e .
```

## 一、Python autotune

由于Runner只是修改了Triton的编译流程，使用`@triton_runner.jit` 替换掉 `@triton.jit`后其是原生支持的。

```Python
@triton.autotune(
    configs=[
        triton.Config({'BT': bt}, num_warps=nw, num_stages=ns)
        for bt in BT_LIST_AUTOTUNE
        for nw in NUM_WARPS_AUTOTUNE
        for ns in [2, 3]
    ],
    key=['H', 'D'],
    **autotune_cache_kwargs,
)
# @triton.jit
@triton_runner.jit
def kda_gate_fwd_kernel(
    g, A, y,
    g_bias,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    T,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
```

示例依赖fla，请`pip install flash-linear-attention`后，切换到triton到`3.5.0`,`pip install triton==3.5.0`。

运行`TRITON_PRINT_AUTOTUNING=1 python examples/autotune/python/test_kda_gate_single.py`后可以看到如下内容

![image](https://img2024.cnblogs.com/blog/1154439/202511/1154439-20251113073951452-636270224.png)

本质上就是暴力尝试了不同的`triton.Config`，做了bench后，并保留了一个cache文件。


## 二、cubin autotune

Triton Runner有接管autotune的计划，但是选择在哪个层级还未决定。

所以在cubin的autotune中，和多层级执行相同，需要指定的是cubin的文件夹位置，其参数为`autotune_cubin_dir`。这里注意其cubin必须是由Triton Runner产生的，其内的metajson会带上形如`"(('g', '*fp32', 'D', False), ('A', '*fp32', 'D', False), ('y', '*fp32', 'D', False), ('g_bias', 'constexpr', None, False), ('beta', 'constexpr', 1.0, False), ('threshold', 'constexpr', 20.0, False), ('T', 'i32', '', False), ('H', 'i32', '', False), ('D', 'constexpr', 12, False), ('BT', 'constexpr', 128, True), ('BD', 'constexpr', 16, True), ('HAS_BIAS', 'constexpr', False, True))"`的`kernel_signature`。

以下代码为示例，全部代码在[gate.py](cubin/gate.py)。

```Python
cache_dir = Path(triton_runner.get_file_dir(__file__)).parent / f"kda_gate_fwd_kernel_cache_sm{capability}"

@triton.autotune(
    configs=[
        triton.Config({'autotune_cubin_dir': str(p)}) for p in cache_dir.iterdir() if p.is_dir()
    ],
    key=['H', 'D'],
)
```

当然`cache_dir`也可以是你的`TRITON_CACHE_DIR`，如出现问题请上报issue。

如果是`sm90`或`sm75`，可以直接运行`TRITON_PRINT_AUTOTUNING=1 python examples/autotune/cubin/test_kda_gate_single.py`示例。
