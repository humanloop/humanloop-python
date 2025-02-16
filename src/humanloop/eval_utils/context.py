from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class EvaluationContext:
    """Context Log to Humanloop.

    Per datapoint state that is set when an Evaluation is ran.
    """

    """Required for associating a Log with the Evaluation Run."""
    source_datapoint_id: str

    """Overloaded .log method call."""
    upload_callback: Callable[[str], None]

    """ID of the evaluated File."""
    file_id: str

    """Path of the evaluated File."""
    path: str

    """Required for associating a Log with the Evaluation Run."""
    run_id: str


EVALUATION_CONTEXT_VARIABLE_NAME = "__EVALUATION_CONTEXT"

_EVALUATION_CONTEXT_VAR: ContextVar[EvaluationContext] = ContextVar(EVALUATION_CONTEXT_VARIABLE_NAME)

_UnsafeEvaluationContextRead = RuntimeError("EvaluationContext not set in the current thread.")


def set_evaluation_context(context: EvaluationContext):
    _EVALUATION_CONTEXT_VAR.set(context)


def get_evaluation_context() -> EvaluationContext:
    try:
        return _EVALUATION_CONTEXT_VAR.get()
    except LookupError:
        raise _UnsafeEvaluationContextRead


def evaluation_context_set() -> bool:
    try:
        _EVALUATION_CONTEXT_VAR.get()
        return True
    except LookupError:
        return False


def log_belongs_to_evaluated_file(log_args: dict[str, Any]) -> bool:
    try:
        evaluation_context: EvaluationContext = _EVALUATION_CONTEXT_VAR.get()
        return evaluation_context.file_id == log_args.get("id") or evaluation_context.path == log_args.get("path")
    except LookupError:
        # Not in an evaluation context
        return False


def is_evaluated_file(file_path) -> bool:
    try:
        evaluation_context = _EVALUATION_CONTEXT_VAR.get()
        return evaluation_context.path == file_path
    except LookupError:
        raise _UnsafeEvaluationContextRead
