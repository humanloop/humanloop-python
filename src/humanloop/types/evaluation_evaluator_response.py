# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
from .evaluator_response import EvaluatorResponse
import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2
import typing


class EvaluationEvaluatorResponse(UncheckedBaseModel):
    version: EvaluatorResponse
    orchestrated: bool = pydantic.Field()
    """
    Whether the Evaluator is orchestrated by Humanloop. Default is `True`. If `False`, a log for the Evaluator should be submitted by the user via the API.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
