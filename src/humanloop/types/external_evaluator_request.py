# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
from .evaluator_arguments_type import EvaluatorArgumentsType
import pydantic
from .evaluator_return_type_enum import EvaluatorReturnTypeEnum
import typing
from .evaluator_judgment_option_response import EvaluatorJudgmentOptionResponse
from .evaluator_judgment_number_limit import EvaluatorJudgmentNumberLimit
from .valence import Valence
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class ExternalEvaluatorRequest(UncheckedBaseModel):
    arguments_type: EvaluatorArgumentsType = pydantic.Field()
    """
    Whether this Evaluator is target-free or target-required.
    """

    return_type: EvaluatorReturnTypeEnum = pydantic.Field()
    """
    The type of the return value of the Evaluator.
    """

    attributes: typing.Optional[typing.Dict[str, typing.Optional[typing.Any]]] = pydantic.Field(default=None)
    """
    Additional fields to describe the Evaluator. Helpful to separate Evaluator versions from each other with details on how they were created or used.
    """

    options: typing.Optional[typing.List[EvaluatorJudgmentOptionResponse]] = pydantic.Field(default=None)
    """
    The options that can be applied as judgments. Only for Evaluators with `return_type` of 'boolean', 'select' or 'multi_select'.
    """

    number_limits: typing.Optional[EvaluatorJudgmentNumberLimit] = pydantic.Field(default=None)
    """
    Limits on the judgment that can be applied. Only for Evaluators with `return_type` of 'number'.
    """

    number_valence: typing.Optional[Valence] = pydantic.Field(default=None)
    """
    The valence of the number judgment. Only for Evaluators with `return_type` of 'number'. If 'positive', a higher number is better. If 'negative', a lower number is better.
    """

    evaluator_type: typing.Literal["external"] = "external"

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
