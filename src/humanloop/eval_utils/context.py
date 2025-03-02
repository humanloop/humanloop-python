from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable
from opentelemetry.trace import Tracer


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

    """Whether a log has been made for this datapoint/ run_id pair."""
    logged: bool


_EVALUATION_CONTEXT_VAR: ContextVar[EvaluationContext] = ContextVar("__EVALUATION_CONTEXT")

_UnsafeContextRead = RuntimeError("Attempting to read from thread Context when variable was not set.")


def set_evaluation_context(context: EvaluationContext):
    _EVALUATION_CONTEXT_VAR.set(context)


def get_evaluation_context() -> EvaluationContext:
    try:
        return _EVALUATION_CONTEXT_VAR.get()
    except LookupError:
        raise _UnsafeContextRead


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
        raise _UnsafeContextRead


@dataclass
class PromptUtilityContext:
    tracer: Tracer
    _in_prompt_utility: int

    @property
    def in_prompt_utility(self) -> bool:
        return self._in_prompt_utility > 0


_PROMPT_UTILITY_CONTEXT_VAR: ContextVar[PromptUtilityContext] = ContextVar("__PROMPT_UTILITY_CONTEXT")


def in_prompt_utility_context() -> bool:
    try:
        return _PROMPT_UTILITY_CONTEXT_VAR.get().in_prompt_utility
    except LookupError:
        return False


def set_prompt_utility_context(tracer: Tracer):
    global _PROMPT_UTILITY_CONTEXT_VAR
    try:
        prompt_utility_context = _PROMPT_UTILITY_CONTEXT_VAR.get()
        # Already set, push another context
        prompt_utility_context._in_prompt_utility += 1
        _PROMPT_UTILITY_CONTEXT_VAR.set(prompt_utility_context)
    except LookupError:
        _PROMPT_UTILITY_CONTEXT_VAR.set(
            PromptUtilityContext(
                tracer=tracer,
                _in_prompt_utility=1,
            )
        )


def get_prompt_utility_context() -> PromptUtilityContext:
    try:
        return _PROMPT_UTILITY_CONTEXT_VAR.get()
    except LookupError:
        raise _UnsafeContextRead


def unset_prompt_utility_context():
    global _PROMPT_UTILITY_CONTEXT_VAR_TOKEN
    try:
        prompt_utility_context = _PROMPT_UTILITY_CONTEXT_VAR.get()
        if prompt_utility_context._in_prompt_utility >= 1:
            prompt_utility_context._in_prompt_utility -= 1
        else:
            raise ValueError("No matching unset_prompt_utility_context() call.")
    except LookupError:
        raise _UnsafeContextRead
