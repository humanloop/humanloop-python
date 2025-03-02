from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Callable, Generator, Optional
from opentelemetry import context as context_api

from humanloop.otel.constants import (
    HUMANLOOP_CONTEXT_EVALUATION,
    HUMANLOOP_CONTEXT_PROMPT,
    HUMANLOOP_CONTEXT_TRACE_ID,
)


def get_trace_id() -> Optional[str]:
    key = hash((HUMANLOOP_CONTEXT_TRACE_ID, threading.get_ident()))
    return context_api.get_value(key=key)


@contextmanager
def set_trace_id(flow_log_id: str) -> Generator[None, None, None]:
    key = hash((HUMANLOOP_CONTEXT_TRACE_ID, threading.get_ident()))
    token = context_api.attach(context_api.set_value(key=key, value=flow_log_id))
    yield
    context_api.detach(token=token)


@dataclass
class PromptContext:
    path: str
    template: Optional[str]


@contextmanager
def set_prompt_context(prompt_context: PromptContext) -> Generator[None, None, None]:
    key = hash((HUMANLOOP_CONTEXT_PROMPT, threading.get_ident()))
    reset_token = context_api.attach(
        context_api.set_value(
            key=key,
            value=prompt_context,
        )
    )
    yield
    context_api.detach(token=reset_token)


def get_prompt_context() -> Optional[PromptContext]:
    key = hash((HUMANLOOP_CONTEXT_PROMPT, threading.get_ident()))
    return context_api.get_value(key)


class EvaluationContext:
    source_datapoint_id: str
    run_id: str
    callback: Callable[[str], None]
    file_id: str
    path: str
    logging_counter: int

    def __init__(
        self,
        source_datapoint_id: str,
        run_id: str,
        callback: Callable[[str], None],
        file_id: str,
        path: str,
    ):
        self.source_datapoint_id = source_datapoint_id
        self.run_id = run_id
        self.callback = callback
        self.file_id = file_id
        self.path = path
        self.logging_counter = 0


@contextmanager
def set_evaluation_context(evaluation_context: EvaluationContext) -> Generator[None, None, None]:
    key = hash((HUMANLOOP_CONTEXT_EVALUATION, threading.get_ident()))
    reset_token = context_api.attach(context_api.set_value(key, evaluation_context))
    yield
    context_api.detach(token=reset_token)


def get_evaluation_context() -> Optional[EvaluationContext]:
    key = hash((HUMANLOOP_CONTEXT_EVALUATION, threading.get_ident()))
    return context_api.get_value(key)
