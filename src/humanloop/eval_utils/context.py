from typing import Callable, TypedDict


class EvaluationContext(TypedDict):
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
