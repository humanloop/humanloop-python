# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions

from .datapoint_response import DatapointResponseParams
from .evaluated_version_response import EvaluatedVersionResponseParams
from .src_external_app_models_v_5_logs_log_response import SrcExternalAppModelsV5LogsLogResponseParams


class EvaluationReportLogResponseParams(typing_extensions.TypedDict):
    evaluated_version: EvaluatedVersionResponseParams
    """
    The version of the Prompt, Tool or Evaluator that the Log belongs to.
    """

    datapoint: DatapointResponseParams
    """
    The Datapoint used to generate the Log
    """

    log: typing_extensions.NotRequired[SrcExternalAppModelsV5LogsLogResponseParams]
    """
    The Log that was evaluated by the Evaluator.
    """

    evaluator_logs: typing.Sequence[SrcExternalAppModelsV5LogsLogResponseParams]
    """
    The Evaluator Logs containing the judgments for the Log.
    """