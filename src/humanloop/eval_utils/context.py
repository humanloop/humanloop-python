import typing
from contextvars import ContextVar
from typing import Callable, TypedDict


class EvaluationContext(TypedDict):
    """Context required by the Exporter when uploading a Log to Humanloop.

    When using the evaluation run utility on decorated functions, the utility
    has does not control the Log upload - the Exporter does. This context class
    propagates the required information to the exporter and allows it to notify
    the utility via a callback.
    """

    """Required for uploading the Log in the Exporter."""
    source_datapoint_id: str

    """Exporter calls this so the eval_utils are notified to evaluate an uploaded Log."""
    upload_callback: Callable[[dict], None]

    """Logs of multiple Files can be uploaded by the Exporter while
    evaluating a single one of them. This identifies the File that
    owns Logs that are part of the Evaluation."""
    evaluated_file_id: str

    """Required for uploading the Log in the Exporter."""
    run_id: str


EVALUATION_CONTEXT: ContextVar[typing.Optional[EvaluationContext]] = ContextVar("__EVALUATION_CONTEXT")
