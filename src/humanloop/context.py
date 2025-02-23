from dataclasses import dataclass
import threading
from typing import Callable, Optional
from opentelemetry import context as context_api

from humanloop.otel.constants import (
    HUMANLOOP_CONTEXT_EVALUATION,
    HUMANLOOP_CONTEXT_PROMPT_PATH,
    HUMANLOOP_CONTEXT_TRACE_ID,
)


ResetToken = object


def get_trace_id() -> Optional[str]:
    key = hash((HUMANLOOP_CONTEXT_TRACE_ID, threading.get_ident()))
    context_api.get_value()
    return context_api.get_value(key=key)


def set_trace_id(flow_log_id: str) -> ResetToken:
    key = hash((HUMANLOOP_CONTEXT_TRACE_ID, threading.get_ident()))
    return context_api.attach(context_api.set_value(key=key, value=flow_log_id))


def reset_trace_id_context(token: ResetToken):
    context_api.detach(token=token)


def set_prompt_path(path: str) -> ResetToken:
    key = hash((HUMANLOOP_CONTEXT_PROMPT_PATH, threading.get_ident()))
    return context_api.set_value(key=key, value=path)


def reset_prompt_path(token: ResetToken):
    context_api.detach(token=token)


def get_prompt_path() -> Optional[str]:
    key = hash((HUMANLOOP_CONTEXT_PROMPT_PATH, threading.get_ident()))
    return context_api.get_value(key)


@dataclass
class EvaluationContext:
    source_datapoint_id: str
    run_id: str
    callback: Callable[[str], None]
    file_id: str
    path: str


def set_evaluation_context(evaluation_context: EvaluationContext):
    key = hash((HUMANLOOP_CONTEXT_EVALUATION, threading.get_ident()))
    context_api.set_value(key, evaluation_context)


def get_evaluation_context() -> Optional[EvaluationContext]:
    key = hash((HUMANLOOP_CONTEXT_EVALUATION, threading.get_ident()))
    return context_api.get_value(key)
