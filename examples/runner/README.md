Example commands for the multi-level runner with Triton **v3.7.0**. For other Triton versions, see the corresponding examples directory.
- For Triton v3.6.0, in [examples/runner/v3.6.0](./../examples/runner/v3.6.0).
- For Triton v3.5.x, in [examples/runner/v3.5.x](./../examples/runner/v3.5.x).
- For Triton v3.4.0, in [examples/runner/v3.4.0](./../examples/runner/v3.4.0).
- For Triton v3.3.1 or v3.3.0, in [examples/runner/v3.3.x](./../examples/runner/v3.3.x).
- For Triton v3.2.0, in [examples/runner/v3.2.0](./../examples/runner/v3.2.0).
- For Triton v3.1.0, in [examples/runner/v3.1.0](./../examples/runner/v3.1.0).
- For Triton v3.0.0, in [examples/runner/v3.0.0](./../examples/runner/v3.0.0).

### sm90 (H100, H200, H20, etc.)
```shell
python examples/runner/python/triton/matmul-with-tma-v4.py

python examples/runner/v3.7.0/ttir/matmul-with-tma/matmul-with-tma-v4.py

python examples/runner/v3.7.0/ttgir/sm90/matmul-with-tma-v4.py

python examples/runner/v3.7.0/llir/sm90/matmul-with-tma-v4.py

python examples/runner/v3.7.0/ptx/sm90/matmul-with-tma-v4.py

python examples/runner/v3.7.0/cubin/sm90/matmul-with-tma-v4.py

python examples/runner/python/gluon/01-intro.py
python examples/runner/python/gluon/02-layouts.py
```

### sm80 (A100, A30)
```shell
python examples/runner/python/triton/matmul-with-dot-v2.py

python examples/runner/v3.7.0/ttir/matmul-with-dot/matmul-with-dot-v2.py

python examples/runner/v3.7.0/ttgir/sm80/matmul-with-dot-v2.py

python examples/runner/v3.7.0/llir/sm80/matmul-with-dot-v2.py

python examples/runner/v3.7.0/ptx/sm80/matmul-with-dot-v2.py

python examples/runner/v3.7.0/cubin/sm80/matmul-with-dot-v2.py

python examples/runner/python/gluon/01-intro.py
python examples/runner/python/gluon/02-layouts.py
```

### sm120 (RTX PRO 6000, RTX 5090, etc.)
```shell
python examples/runner/python/triton/matmul-with-tma-v4.py

python examples/runner/v3.7.0/ttir/matmul-with-tma/matmul-with-tma-v4.py

python examples/runner/v3.7.0/ttgir/sm120/matmul-with-tma-v4.py

python examples/runner/v3.7.0/llir/sm120/matmul-with-tma-v4.py

python examples/runner/v3.7.0/ptx/sm120/matmul-with-tma-v4.py

python examples/runner/v3.7.0/cubin/sm120/matmul-with-tma-v4.py

python examples/runner/python/gluon/01-intro.py
python examples/runner/python/gluon/02-layouts.py
```

### sm86 (A10, RTX 3090, etc.)
```shell
python examples/runner/python/triton/matmul-with-dot-v2.py

python examples/runner/v3.7.0/ttir/matmul-with-dot/matmul-with-dot-v2.py

python examples/runner/v3.7.0/ttgir/sm86/matmul-with-dot-v2.py

python examples/runner/v3.7.0/llir/sm86/matmul-with-dot-v2.py

python examples/runner/v3.7.0/ptx/sm86/matmul-with-dot-v2.py

python examples/runner/v3.7.0/cubin/sm86/matmul-with-dot-v2.py

python examples/runner/python/gluon/01-intro.py
python examples/runner/python/gluon/02-layouts.py
```

### sm75 (T4, RTX 2080, etc.)
```shell
python examples/runner/python/triton/matmul.py

python examples/runner/v3.7.0/ttir/matmul/matmul.py

python examples/runner/v3.7.0/ttgir/sm75/matmul.py

python examples/runner/v3.7.0/llir/sm75/matmul.py

python examples/runner/v3.7.0/ptx/sm75/matmul.py

python examples/runner/v3.7.0/cubin/sm75/matmul.py

python examples/runner/python/gluon/01-intro.py
python examples/runner/python/gluon/02-layouts.py
```

### AMD CDNA3 (MI300 series)

```shell
python examples/runner/amd/v3.6.0/ttir/matmul.py

python examples/runner/amd/v3.6.0/ttgir/matmul.py

python examples/runner/amd/v3.6.0/llir/matmul.py

python examples/runner/amd/v3.6.0/amdgcn/matmul.py

python examples/runner/amd/v3.6.0/hsaco/matmul.py
```
