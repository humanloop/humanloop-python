from typing import TypedDict, Callable
import typing
from contextvars import ContextVar


class EvaluationContext(TypedDict):
    source_datapoint_id: str
    upload_callback: Callable[[dict], None]
    evaluated_file_id: str
    run_id: str


EVALUATION_CONTEXT: ContextVar[typing.Optional[EvaluationContext]] = ContextVar("__EVALUATION_CONTEXT")
