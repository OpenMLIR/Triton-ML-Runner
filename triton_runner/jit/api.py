from typing import Callable, Iterable, Optional, Union, overload

from triton.runtime.jit import T

from .versions import (
    RunnerJITFunction,
    RunnerJITFunction_TLX,
    RunnerJITFunctionV3_8_0,
    RunnerJITFunctionV3_7_0,
    RunnerJITFunctionV3_6_0,
    RunnerJITFunctionV3_5_0,
    RunnerJITFunctionV3_4_0,
    RunnerJITFunctionV3_3_0,
    RunnerJITFunctionV3_2_0,
    RunnerJITFunctionV3_1_0,
)


@overload
def jit(fn: T) -> RunnerJITFunction[T]:
    ...


@overload
def jit(
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[T], RunnerJITFunction[T]]:
    ...


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[RunnerJITFunction[T], Callable[[T], RunnerJITFunction[T]]]:

    def decorator(fn: T) -> RunnerJITFunction[T]:
        assert callable(fn)
        from ..compat.version import is_tlx
        from ..compat.version import (
            is_triton_v3_8,
            is_triton_v3_7,
            is_triton_v3_6,
            is_triton_v3_5,
            is_triton_v3_4,
            is_triton_v3_3,
            is_triton_v3_2,
            is_triton_v3_1,
            is_triton_v3_0,
        )
        from ..compat.version import triton_version

        common_kwargs = {
            "fn": fn,
            "version": version,
            "do_not_specialize": do_not_specialize,
            "debug": debug,
            "noinline": noinline,
            "repr": repr,
            "launch_metadata": launch_metadata,
        }

        dispatch_map = [
            (is_tlx, RunnerJITFunction_TLX),
            (is_triton_v3_8, RunnerJITFunctionV3_8_0),
            (is_triton_v3_7, RunnerJITFunctionV3_7_0),
            (is_triton_v3_6, RunnerJITFunctionV3_6_0),
            (is_triton_v3_5, RunnerJITFunctionV3_5_0),
            (is_triton_v3_4, RunnerJITFunctionV3_4_0),
            (is_triton_v3_3, RunnerJITFunctionV3_3_0),
            (is_triton_v3_2, RunnerJITFunctionV3_2_0),
            (is_triton_v3_1 or is_triton_v3_0, RunnerJITFunctionV3_1_0),
        ]

        for condition, cls in dispatch_map:
            if condition:
                if cls is not RunnerJITFunctionV3_1_0:
                    common_kwargs["do_not_specialize_on_alignment"] = do_not_specialize_on_alignment

                return cls(**common_kwargs)
        raise RuntimeError(f"@triton_runner.jit doesn't support Triton v{triton_version}.")

    if fn is not None:
        return decorator(fn)
    else:
        return decorator
