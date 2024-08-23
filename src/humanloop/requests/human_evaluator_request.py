# This file was auto-generated by Fern from our API Definition.

import typing_extensions
from ..types.evaluator_arguments_type import EvaluatorArgumentsType
from ..types.human_evaluator_request_return_type import HumanEvaluatorRequestReturnType
import typing
import typing_extensions
from .evaluator_judgment_option_response import EvaluatorJudgmentOptionResponseParams
from .evaluator_judgment_number_limit import EvaluatorJudgmentNumberLimitParams
from ..types.valence import Valence


class HumanEvaluatorRequestParams(typing_extensions.TypedDict):
    arguments_type: EvaluatorArgumentsType
    """
    Whether this evaluator is target-free or target-required.
    """

    return_type: HumanEvaluatorRequestReturnType
    """
    The type of the return value of the Evaluator.
    """

    evaluator_type: typing.Literal["human"]
    instructions: typing_extensions.NotRequired[str]
    """
    Instructions for the Human annotating the .
    """

    options: typing_extensions.NotRequired[typing.Sequence[EvaluatorJudgmentOptionResponseParams]]
    """
    The options that the Human annotator can choose from.
    """

    number_limits: typing_extensions.NotRequired[EvaluatorJudgmentNumberLimitParams]
    """
    Limits on the judgment that can be applied. Only for Evaluators with `return_type` of `'number'`.
    """

    number_valence: typing_extensions.NotRequired[Valence]
    """
    The valence of the number judgment. Only for Evaluators with `return_type` of `'number'`. If 'positive', a higher number is better. If 'negative', a lower number is better.
    """
