from typing import Callable, TypedDict


class EvaluationContext(TypedDict):
    """Context Log to Humanloop.

    Global state that is set when an Evaluation is ran.
    """

    """Required for associating a Log with the Evaluation Run."""
    source_datapoint_id: str

    """Exporter calls this so the eval_utils are notified to evaluate an uploaded Log."""
    upload_callback: Callable[[dict], None]

    """ID of the evaluated File."""
    file_id: str

    """Path of the evaluated File."""
    path: str

    """Required for associating a Log with the Evaluation Run."""
    run_id: str


EVALUATION_CONTEXT_VARIABLE_NAME = "__EVALUATION_CONTEXT"
