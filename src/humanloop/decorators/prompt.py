import logging
import typing
from functools import wraps
from typing import Awaitable, Callable, Literal, TypeVar, Union, overload

from typing_extensions import ParamSpec

from humanloop.context import DecoratorContext, set_decorator_context
from humanloop.evals.types import FileEvalConfig

logger = logging.getLogger("humanloop.sdk")

P = ParamSpec("P")
R = TypeVar("R")


def prompt_decorator_factory(path: str):
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        wrapper = _wrapper_factory(
            func=func,
            path=path,
            is_awaitable=False,
        )

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
        wrapper = _wrapper_factory(
            func=func,
            path=path,
            is_awaitable=True,
        )

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


@overload
def _wrapper_factory(
    func: Callable[P, Awaitable[R]],
    path: str,
    is_awaitable: Literal[True],
) -> Callable[P, Awaitable[R]]: ...


@overload
def _wrapper_factory(
    func: Callable[P, R],
    path: str,
    is_awaitable: Literal[False],
) -> Callable[P, R]: ...


def _wrapper_factory(  # type: ignore [misc]
    func: Union[Callable[P, Awaitable[R]], Callable[P, R]],
    path: str,
    is_awaitable: bool,
):
    """Create a wrapper function for a prompt-decorated function.

    Args:
        func: The function to decorate
        path: The path to the prompt
        is_awaitable: Whether the function is an async function

    Returns:
        A wrapper function that sets up the decorator context
    """
    if is_awaitable:
        func = typing.cast(Callable[P, Awaitable[R]], func)

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
                output = await func(*args, **kwargs)  # type: ignore [misc]
                return output  # type: ignore [return-value]
    else:
        func = typing.cast(Callable[P, R], func)

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
                output = func(*args, **kwargs)  # type: ignore [misc]
                return output  # type: ignore [return-value]

    return wrapper
