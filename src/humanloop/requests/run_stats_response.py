# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import typing_extensions
import typing
from .run_stats_response_evaluator_stats_item import (
    RunStatsResponseEvaluatorStatsItemParams,
)
from ..types.evaluation_status import EvaluationStatus


class RunStatsResponseParams(typing_extensions.TypedDict):
    """
    Stats for a Run in the Evaluation.
    """

    run_id: str
    """
    Unique identifier for the Run.
    """

    version_id: typing_extensions.NotRequired[str]
    """
    Unique identifier for the evaluated Version.
    """

    batch_id: typing_extensions.NotRequired[str]
    """
    Unique identifier for the batch of Logs to include in the Evaluation.
    """

    num_logs: int
    """
    The total number of existing Logs in this Run.
    """

    evaluator_stats: typing.Sequence[RunStatsResponseEvaluatorStatsItemParams]
    """
    Stats for each Evaluator Version applied to this Run.
    """

    status: EvaluationStatus
    """
    The current status of the Run.
    """
