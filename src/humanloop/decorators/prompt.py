from functools import wraps
import logging

from typing_extensions import ParamSpec
from typing import Callable, TypeVar

from humanloop.context import DecoratorContext, set_decorator_context
from humanloop.evals.types import File

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

        wrapper.file = File(  # type: ignore [attr-defined]
            path=path,
            type="prompt",
            version={  # type: ignore [typeddict-item]
                "template": None,
            },
            callable=wrapper,
        )

        return wrapper

    return decorator
