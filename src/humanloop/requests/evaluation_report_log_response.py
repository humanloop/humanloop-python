# This file was auto-generated by Fern from our API Definition.

import typing_extensions
from .evaluated_version_response import EvaluatedVersionResponseParams
from .datapoint_response import DatapointResponseParams
import typing_extensions
from .log_response import LogResponseParams
import typing


class EvaluationReportLogResponseParams(typing_extensions.TypedDict):
    evaluated_version: EvaluatedVersionResponseParams
    """
    The version of the Prompt, Tool or Evaluator that the Log belongs to.
    """

    datapoint: DatapointResponseParams
    """
    The Datapoint used to generate the Log
    """

    log: typing_extensions.NotRequired[LogResponseParams]
    """
    The Log that was evaluated by the Evaluator.
    """

    evaluator_logs: typing.Sequence[LogResponseParams]
    """
    The Evaluator Logs containing the judgments for the Log.
    """
