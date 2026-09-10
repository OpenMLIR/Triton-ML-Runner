from .versions import (
    RunnerJITFunction,
    RunnerJITFunctionV3_8_0,
    RunnerJITFunctionV3_7_0,
    RunnerJITFunctionV3_4_0,
    RunnerJITFunctionV3_5_0,
    RunnerJITFunctionV3_6_0,
)
from triton.runtime.jit import JITFunction
from triton.experimental.gluon._runtime import GluonASTSource, T
from typing import Optional, Callable, Iterable, Union


class RunnerGluonJITFunction(RunnerJITFunction[T]):
    pass

def make_gluon_runner(base_cls):
    class GluonRunner(base_cls):
        def create_binder(self):
            result = super().create_binder()
            self.ASTSource = GluonASTSource
            return result

        def is_gluon(self):
            return True

    GluonRunner.__name__ = base_cls.__name__.replace("RunnerJIT", "RunnerGluonJIT")
    return GluonRunner

RunnerGluonJITFunctionV3_8_0 = make_gluon_runner(RunnerJITFunctionV3_8_0[T])
RunnerGluonJITFunctionV3_7_0 = make_gluon_runner(RunnerJITFunctionV3_7_0[T])
RunnerGluonJITFunctionV3_6_0 = make_gluon_runner(RunnerJITFunctionV3_6_0[T])
RunnerGluonJITFunctionV3_5_0 = make_gluon_runner(RunnerJITFunctionV3_5_0[T])
RunnerGluonJITFunctionV3_4_0 = make_gluon_runner(RunnerJITFunctionV3_4_0[T])


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[RunnerGluonJITFunction[T], Callable[[T], JITFunction[T]]]:
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, arguments are
        implicitly converted to pointers if they have a :code:`.data_ptr()` method
        and a `.dtype` attribute.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * builtins within the triton package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """

    def decorator(fn: T) -> JITFunction[T]:
        assert callable(fn)
        from ..compat.version import is_triton_v3_8, is_triton_v3_7, is_triton_v3_6, is_triton_v3_5, is_triton_v3_4, triton_version

        kwargs = {
            "fn": fn,
            "version": version,
            "do_not_specialize": do_not_specialize,
            "do_not_specialize_on_alignment": do_not_specialize_on_alignment,
            "debug": debug,
            "noinline": noinline,
            "repr": repr,
            "launch_metadata": launch_metadata,
        }

        dispatch_map = [
            (is_triton_v3_8, RunnerGluonJITFunctionV3_8_0),
            (is_triton_v3_7, RunnerGluonJITFunctionV3_7_0),
            (is_triton_v3_6, RunnerGluonJITFunctionV3_6_0),
            (is_triton_v3_5, RunnerGluonJITFunctionV3_5_0),
            (is_triton_v3_4, RunnerGluonJITFunctionV3_4_0),
        ]

        for condition, runner_cls in dispatch_map:
            if condition:
                return runner_cls(**kwargs)

        raise RuntimeError(f"@gluon_runner.jit doesn't support Triton v{triton_version}.")

    if fn is not None:
        return decorator(fn)

    else:
        return decorator
