import triton

triton_version = triton.__version__
_parts = triton_version.split('.')
_ver = (int(_parts[0]), int(_parts[1]))

# Supported range: 3.0 ~ 3.8 inclusive
is_support_version = (3, 0) <= _ver <= (3, 8)

is_triton_v3_8 = _ver == (3, 8)
is_triton_v3_7 = _ver == (3, 7)
is_triton_v3_6 = _ver == (3, 6)
is_triton_v3_5 = _ver == (3, 5)
is_triton_v3_4 = _ver == (3, 4)
is_triton_v3_3 = _ver == (3, 3)
is_triton_v3_2 = _ver == (3, 2)
is_triton_v3_1 = _ver == (3, 1)
is_triton_v3_0 = _ver == (3, 0)

is_triton_geq_v3_3 = _ver >= (3, 3)
is_triton_geq_v3_4 = _ver >= (3, 4)
is_triton_geq_v3_5 = _ver >= (3, 5)

is_triton_leq_v3_2 = _ver <= (3, 2)
is_triton_leq_v3_1 = _ver <= (3, 1)

is_disable_multithreading = (3, 2) <= _ver <= (3, 5)

if is_triton_v3_5:
    uni_triton_version = "3.5.x"
elif is_triton_v3_3:
    uni_triton_version = "3.3.x"
else:
    uni_triton_version = triton_version

try:
    import triton.language.extra.tlx as tlx
    is_tlx = True
except ImportError:
    is_tlx = False
