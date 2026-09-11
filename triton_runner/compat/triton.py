"""Version-conditional triton imports, unified across all supported versions."""
from .version import is_triton_geq_v3_4, is_triton_geq_v3_5, is_tlx_v3_7_4

# triton_key moved from triton.compiler.compiler to triton.runtime.cache in 3.5
if is_triton_geq_v3_5 or is_tlx_v3_7_4:
    from triton.runtime.cache import triton_key
else:
    from triton.compiler.compiler import triton_key


def get_triton_cache_dir() -> str:
    """Return the active Triton cache directory, compatible across versions."""
    import os
    if is_triton_geq_v3_4:
        from triton import knobs
        return knobs.cache.dir
    else:
        from triton.runtime.cache import default_cache_dir
        return os.getenv("TRITON_CACHE_DIR", "").strip() or default_cache_dir()


__all__ = ["triton_key", "get_triton_cache_dir"]
