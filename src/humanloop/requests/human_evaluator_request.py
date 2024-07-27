# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions

from ..types.evaluator_arguments_type import EvaluatorArgumentsType
from ..types.evaluator_return_type_enum import EvaluatorReturnTypeEnum


class HumanEvaluatorRequestParams(typing_extensions.TypedDict):
    arguments_type: EvaluatorArgumentsType
    """
    Whether this evaluator is target-free or target-required.
    """

    return_type: EvaluatorReturnTypeEnum
    """
    The type of the return value of the evaluator.
    """

    evaluator_type: typing.Literal["human"]
