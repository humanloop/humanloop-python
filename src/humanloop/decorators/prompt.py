import logging
from functools import wraps
from typing import Awaitable, Callable, TypeVar

from typing_extensions import ParamSpec

from humanloop.context import DecoratorContext, set_decorator_context
from humanloop.evals.types import FileEvalConfig

logger = logging.getLogger("humanloop.sdk")

P = ParamSpec("P")
R = TypeVar("R")


def prompt_decorator_factory(path: str):
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with set_decorator_context(
                DecoratorContext(
                    path=path,
                    type="prompt",
                    version={
                        # TODO: Implement a reverse-lookup of the template
                        "template": None,
                    },
                )
            ):
                output = func(*args, **kwargs)
                return output

        wrapper.file = FileEvalConfig(  # type: ignore [attr-defined]
            path=path,
            type="prompt",
            version={  # type: ignore [typeddict-item]
                "template": None,
            },
            callable=wrapper,
        )

        return wrapper

    return decorator


def a_prompt_decorator_factory(path: str):
    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with set_decorator_context(
                DecoratorContext(
                    path=path,
                    type="prompt",
                    version={
                        # TODO: Implement a reverse-lookup of the template
                        "template": None,
                    },
                )
            ):
                output = await func(*args, **kwargs)
                return output

        wrapper.file = FileEvalConfig(  # type: ignore [attr-defined]
            path=path,
            type="prompt",
            version={  # type: ignore [typeddict-item]
                "template": None,
            },
            callable=wrapper,
        )

        return wrapper

    return decorator
