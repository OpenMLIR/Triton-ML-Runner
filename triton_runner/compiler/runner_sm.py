from contextlib import contextmanager

from triton.backends.compiler import GPUTarget
from triton.runtime.driver import driver


RUNNER_SM_KWARG = "runner_sm"


def normalize_runner_sm(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("runner_sm must be an int like 90 or a string like 'sm90'")
    if isinstance(value, str):
        value = value.strip().lower()
        if value.startswith("sm"):
            value = value[2:]
        if not value.isdigit():
            raise ValueError("runner_sm must be an int like 90 or a string like 'sm90'")
        sm = int(value)
    elif isinstance(value, int):
        sm = value
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        major, minor = value
        if not isinstance(major, int) or not isinstance(minor, int):
            raise ValueError("runner_sm tuple/list values must be integers")
        sm = major * 10 + minor
    else:
        raise ValueError("runner_sm must be an int like 90 or a string like 'sm90'")

    if sm <= 0:
        raise ValueError("runner_sm must be a positive SM capability")
    return sm


def runner_sm_arch(sm):
    return f"sm{sm}"


def target_with_runner_sm(target, runner_sm):
    sm = normalize_runner_sm(runner_sm)
    if sm is None:
        return target
    if not isinstance(target, GPUTarget) or target.backend != "cuda":
        raise ValueError("runner_sm is only supported for CUDA GPUTarget compilation")
    return GPUTarget(target.backend, sm, getattr(target, "warp_size", 32))


def target_backend_with_runner_sm(target, backend, runner_sm):
    compile_target = target_with_runner_sm(target, runner_sm)
    if runner_sm is None:
        return compile_target, backend
    from triton.compiler.compiler import make_backend
    return compile_target, make_backend(compile_target)


def kwargs_with_runner_sm_arch(kwargs):
    runner_sm = normalize_runner_sm(kwargs.get(RUNNER_SM_KWARG))
    if runner_sm is None:
        return kwargs

    arch = kwargs.get("arch")
    if arch is not None and normalize_runner_sm(arch) != runner_sm:
        raise ValueError(f"runner_sm={runner_sm} conflicts with arch={arch!r}")

    compile_kwargs = dict(kwargs)
    compile_kwargs.pop(RUNNER_SM_KWARG, None)
    compile_kwargs["arch"] = runner_sm_arch(runner_sm)
    return compile_kwargs


@contextmanager
def override_current_target(target, enabled=True):
    if not enabled:
        yield
        return

    original = driver.active.get_current_target
    driver.active.get_current_target = lambda: target
    try:
        yield
    finally:
        driver.active.get_current_target = original


def native_compile_with_runner_sm(src, ast_src, metadata_json, *, target, options, source_path=None, kernel_signature=None, start_pass=None, runner_sm=None):
    from .compile import native_compile
    with override_current_target(target, enabled=runner_sm is not None):
        return native_compile(src, ast_src, metadata_json, target=target, options=options, source_path=source_path, kernel_signature=kernel_signature, start_pass=start_pass)
