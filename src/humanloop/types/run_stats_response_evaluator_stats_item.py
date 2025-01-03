# This file was auto-generated by Fern from our API Definition.

import typing
from .numeric_evaluator_stats_response import NumericEvaluatorStatsResponse
from .boolean_evaluator_stats_response import BooleanEvaluatorStatsResponse
from .select_evaluator_stats_response import SelectEvaluatorStatsResponse
from .text_evaluator_stats_response import TextEvaluatorStatsResponse

RunStatsResponseEvaluatorStatsItem = typing.Union[
    NumericEvaluatorStatsResponse,
    BooleanEvaluatorStatsResponse,
    SelectEvaluatorStatsResponse,
    TextEvaluatorStatsResponse,
]
