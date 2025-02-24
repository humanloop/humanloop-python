import logging


from contextlib import contextmanager
from typing import Optional

from humanloop.context import PromptContext, reset_prompt_context, set_prompt_context

logger = logging.getLogger("humanloop.sdk")


@contextmanager
def prompt(path: str, template: Optional[str]):
    try:
        token = set_prompt_context(
            PromptContext(
                path=path,
                template=template,
            )
        )
        yield
    finally:
        reset_prompt_context(token=token)
