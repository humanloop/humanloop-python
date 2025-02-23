import logging


from contextlib import contextmanager

from humanloop.context import reset_prompt_path, set_prompt_path

logger = logging.getLogger("humanloop.sdk")


@contextmanager
def prompt(path: str):
    try:
        token = set_prompt_path(path=path)
        yield
    finally:
        reset_prompt_path(token=token)
