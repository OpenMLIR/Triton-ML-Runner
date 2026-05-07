__version__ = '0.3.8'

from .compat.version import is_support_version, is_triton_v3_7, triton_version
if not is_support_version:
    raise RuntimeError(f"Triton Runner doesn't support Triton v{triton_version}")


def _env_flag(name, default=True):
    import os
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no", ""}


def _init_is_cuda():
    from triton.runtime import driver
    try:
        target = driver.active.get_current_target()
    except Exception:
        return False
    return getattr(target, "backend", None) == "cuda"


IS_CUDA = _init_is_cuda()

TRITON_RUNNER_PROD_TEST = _env_flag("TRITON_RUNNER_PROD_TEST", default=False)
TRITON_RUNNER_PROD = _env_flag("TRITON_RUNNER_PROD", default=False) or TRITON_RUNNER_PROD_TEST


from .compat.version import is_triton_geq_v3_4
if is_triton_geq_v3_4:
    from .runtime.autotune import autotune

if TRITON_RUNNER_PROD and IS_CUDA and is_triton_v3_7:
    from .tvm_ffi import _require_tvm_ffi
    _require_tvm_ffi()
    from .jit.prod import jit
    from .debug.console import blue_print
    blue_print("[Triton Runner] Production mode enabled")
else:
    from .jit import jit
from .debug import console as color_print
from .runtime import torch as torch_utils
from .runtime.triton_backend import (
    configure_autotune_backend,
    configure_jit_backend,
    restore_autotune_backend,
    restore_jit_backend,
)


def get_file_dir(file):
    import os
    return os.path.dirname(os.path.abspath(file))
