# This file was auto-generated by Fern from our API Definition.

import typing_extensions
from .overall_stats import OverallStatsParams
import typing
from .version_stats_response import VersionStatsResponseParams
import typing_extensions
from ..types.evaluation_status import EvaluationStatus


class EvaluationStatsParams(typing_extensions.TypedDict):
    overall_stats: OverallStatsParams
    """
    Stats for the Evaluation Report as a whole.
    """

    version_stats: typing.Sequence[VersionStatsResponseParams]
    """
    Stats for each Evaluated Version in the Evaluation Report.
    """

    progress: typing_extensions.NotRequired[str]
    """
    A summary string report of the Evaluation's progress you can print to the command line;helpful when integrating Evaluations with CI/CD.
    """

    report: typing_extensions.NotRequired[str]
    """
    A summary string report of the Evaluation you can print to command line;helpful when integrating Evaluations with CI/CD.
    """

    status: EvaluationStatus
    """
    The current status of the Evaluation.
    """
