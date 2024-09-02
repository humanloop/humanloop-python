# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
from .evaluator_arguments_type import EvaluatorArgumentsType
import pydantic
from .evaluator_return_type_enum import EvaluatorReturnTypeEnum
import typing
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class ExternalEvaluatorRequest(UncheckedBaseModel):
    arguments_type: EvaluatorArgumentsType = pydantic.Field()
    """
    Whether this evaluator is target-free or target-required.
    """

    return_type: EvaluatorReturnTypeEnum = pydantic.Field()
    """
    The type of the return value of the evaluator.
    """

    attributes: typing.Optional[typing.Dict[str, typing.Optional[typing.Any]]] = pydantic.Field(default=None)
    """
    Additional fields to describe the Evaluator. Helpful to separate Evaluator versions from each other with details on how they were created or used.
    """

    evaluator_type: typing.Literal["external"] = "external"

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
