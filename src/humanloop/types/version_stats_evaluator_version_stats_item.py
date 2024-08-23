# This file was auto-generated by Fern from our API Definition.

import typing
from .numeric_evaluator_version_stats import NumericEvaluatorVersionStats
from .boolean_evaluator_version_stats import BooleanEvaluatorVersionStats
from .select_evaluator_version_stats import SelectEvaluatorVersionStats
from .text_evaluator_version_stats import TextEvaluatorVersionStats

VersionStatsEvaluatorVersionStatsItem = typing.Union[
    NumericEvaluatorVersionStats, BooleanEvaluatorVersionStats, SelectEvaluatorVersionStats, TextEvaluatorVersionStats
]
