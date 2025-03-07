from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Any, Callable, Generator, Literal, Optional
from opentelemetry import context as context_api

from humanloop.error import HumanloopRuntimeError
from humanloop.otel.constants import (
    HUMANLOOP_CONTEXT_EVALUATION,
    HUMANLOOP_CONTEXT_DECORATOR,
    HUMANLOOP_CONTEXT_TRACE_ID,
)


def get_trace_id() -> Optional[str]:
    key = str(hash((HUMANLOOP_CONTEXT_TRACE_ID, threading.get_ident())))
    return context_api.get_value(key=key)  # type: ignore [return-value]


@contextmanager
def set_trace_id(flow_log_id: str) -> Generator[None, None, None]:
    key = str(hash((HUMANLOOP_CONTEXT_TRACE_ID, threading.get_ident())))
    token = context_api.attach(context_api.set_value(key=key, value=flow_log_id))
    yield
    context_api.detach(token=token)


@dataclass
class DecoratorContext:
    path: str
    type: Literal["prompt", "tool", "flow"]
    version: dict[str, Any]


@contextmanager
def set_decorator_context(
    decorator_context: DecoratorContext,
) -> Generator[DecoratorContext, None, None]:
    key = str(hash((HUMANLOOP_CONTEXT_DECORATOR, threading.get_ident())))
    reset_token = context_api.attach(
        context_api.set_value(
            key=key,
            value=decorator_context,
        )
    )
    yield decorator_context
    context_api.detach(token=reset_token)


def get_decorator_context() -> Optional[DecoratorContext]:
    key = str(hash((HUMANLOOP_CONTEXT_DECORATOR, threading.get_ident())))
    return context_api.get_value(key)  # type: ignore [return-value]


class EvaluationContext:
    source_datapoint_id: str
    run_id: str
    file_id: str
    path: str
    _logged: bool
    _callback: Callable[[str], None]
    _context_log_belongs_eval_file: bool

    @property
    def logged(self) -> bool:
        return self._logged

    @contextmanager
    def spy_log_args(
        self,
        log_args: dict[str, Any],
        path: Optional[str] = None,
        file_id: Optional[str] = None,
    ) -> Generator[dict[str, Any], None, None]:
        if path is None and file_id is None:
            raise HumanloopRuntimeError(
                "Internal error: Evaluation context called without providing a path of file_id"
            )

        if self.path is not None and self.path == path:
            self._logged = True
            self._context_log_belongs_eval_file = True
            yield {
                **log_args,
                "source_datapoint_id": self.source_datapoint_id,
                "run_id": self.run_id,
            }
        elif self.file_id is not None and self.file_id == file_id:
            self._logged = True
            self._context_log_belongs_eval_file = True
            yield {
                **log_args,
                "source_datapoint_id": self.source_datapoint_id,
                "run_id": self.run_id,
            }
        else:
            yield log_args
        self._context_log_belongs_eval_file = False

    @property
    def callback(self) -> Optional[Callable[[str], None]]:
        if self._context_log_belongs_eval_file:
            return self._callback
        return None

    def __init__(
        self,
        source_datapoint_id: str,
        run_id: str,
        eval_callback: Callable[[str], None],
        file_id: str,
        path: str,
    ):
        self.source_datapoint_id = source_datapoint_id
        self.run_id = run_id
        self._callback = eval_callback
        self.file_id = file_id
        self.path = path
        self._logged = False


@contextmanager
def set_evaluation_context(
    evaluation_context: EvaluationContext,
) -> Generator[None, None, None]:
    key = str(hash((HUMANLOOP_CONTEXT_EVALUATION, threading.get_ident())))
    reset_token = context_api.attach(context_api.set_value(key, evaluation_context))
    yield
    context_api.detach(token=reset_token)


def get_evaluation_context() -> Optional[EvaluationContext]:
    key = str(hash((HUMANLOOP_CONTEXT_EVALUATION, threading.get_ident())))
    return context_api.get_value(key)  # type: ignore [return-value]
