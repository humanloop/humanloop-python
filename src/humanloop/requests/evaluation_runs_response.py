# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import typing
from .evaluation_run_response import EvaluationRunResponseParams


class EvaluationRunsResponseParams(typing_extensions.TypedDict):
    runs: typing.Sequence[EvaluationRunResponseParams]
    """
    The Runs in the Evaluation.
    """