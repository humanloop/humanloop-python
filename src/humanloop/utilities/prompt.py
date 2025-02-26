from functools import wraps
import logging


from typing import Callable, Optional

from humanloop.context import PromptContext, reset_prompt_context, set_prompt_context

logger = logging.getLogger("humanloop.sdk")


def prompt_decorator_factory(path: str, template: Optional[str]):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            token = set_prompt_context(
                PromptContext(
                    path=path,
                    template=template,
                )
            )
            output = func(*args, **kwargs)
            reset_prompt_context(token=token)
            return output

        return wrapper

    return decorator
