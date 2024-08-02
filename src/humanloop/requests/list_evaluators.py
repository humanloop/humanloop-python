# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions

from .evaluator_response import EvaluatorResponseParams


class ListEvaluatorsParams(typing_extensions.TypedDict):
    records: typing.Sequence[EvaluatorResponseParams]
    """
    The list of Evaluators.
    """