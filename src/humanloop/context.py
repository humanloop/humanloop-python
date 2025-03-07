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
    # Use threading.get_ident() to ensure the context is unique to the current thread
    key = str(hash((HUMANLOOP_CONTEXT_TRACE_ID, threading.get_ident())))
    return context_api.get_value(key=key)  # type: ignore [return-value]


@contextmanager
def set_trace_id(flow_log_id: str) -> Generator[None, None, None]:
    # Use threading.get_ident() to ensure the context is unique to the current thread
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
    # Use threading.get_ident() to ensure the context is unique to the current thread
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
    # Use threading.get_ident() to ensure the context is unique to the current thread
    key = str(hash((HUMANLOOP_CONTEXT_DECORATOR, threading.get_ident())))
    return context_api.get_value(key)  # type: ignore [return-value]


class EvaluationContext:
    """
    Represents the context for evaluating a specific data point within a run.

    This class integrates with the OpenTelemetry (OTEL) runtime context API to distribute data points across threads.
    Each thread processes one data point by invoking a callable and subsequently logs the results against a run.

    Attributes:
        source_datapoint_id (str): The unique identifier of the source data point.
        run_id (str): The unique identifier of the evaluation run.
        file_id (str): The identifier of the file associated with the evaluation.
        path (str): The file path associated with the evaluation.
        _logged (bool): Tracks whether logging has already occurred in this context to ensure only the first log is counted.
        _callback (Callable[[str], None]): A callback function to be executed when logging occurs.
    """

    source_datapoint_id: str
    run_id: str
    file_id: str
    path: str
    _logged: bool
    _callback: Callable[[str], None]

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

    @property
    def logged(self) -> bool:
        """
        Return true if the current datapoint has been evaluated already.
        """
        return self._logged

    def log_args_with_context(
        self,
        log_args: dict[str, Any],
        path: Optional[str] = None,
        file_id: Optional[str] = None,
    ) -> tuple[dict[str, Any], Optional[Callable[[str], None]]]:
        """
        Logs arguments within the evaluation context if the path or file ID matches.

        This method ensures that if multiple logs are made against the same file, only the first one
        is considered toward the evaluation run. If a log has already been made, subsequent calls
        will return the log arguments without adding evaluation-specific metadata.

        Args:
            log_args (dict[str, Any]): The log arguments to be recorded.
            path (Optional[str]): The file path for logging (if applicable).
            file_id (Optional[str]): The file ID for logging (if applicable).

        Returns:
            tuple[dict[str, Any], Optional[Callable[[str], None]]]:
                - Updated log arguments with additional context information if applicable.
                - A callback function if logging belongs to the evaluation file, otherwise None.

        Raises:
            HumanloopRuntimeError: If neither `path` nor `file_id` is provided.
        """
        if path is None and file_id is None:
            raise HumanloopRuntimeError("Internal error: Evaluation context called without providing a path or file_id")

        # Ensure only the first log against the same file is considered
        if self._logged:
            return log_args, None

        if self.path is not None and self.path == path:
            self._logged = True
            return {
                **log_args,
                "source_datapoint_id": self.source_datapoint_id,
                "run_id": self.run_id,
            }, self._callback
        elif self.file_id is not None and self.file_id == file_id:
            self._logged = True
            return {
                **log_args,
                "source_datapoint_id": self.source_datapoint_id,
                "run_id": self.run_id,
            }, self._callback
        else:
            return log_args, None


@contextmanager
def set_evaluation_context(
    evaluation_context: EvaluationContext,
) -> Generator[None, None, None]:
    # Use threading.get_ident() to ensure the context is unique to the current thread
    key = str(hash((HUMANLOOP_CONTEXT_EVALUATION, threading.get_ident())))
    reset_token = context_api.attach(context_api.set_value(key, evaluation_context))
    yield
    context_api.detach(token=reset_token)


def get_evaluation_context() -> Optional[EvaluationContext]:
    # Use threading.get_ident() to ensure the context is unique to the current thread
    key = str(hash((HUMANLOOP_CONTEXT_EVALUATION, threading.get_ident())))
    return context_api.get_value(key)  # type: ignore [return-value]
