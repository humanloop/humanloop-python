# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
import typing
import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class EvaluatorJudgmentNumberLimit(UncheckedBaseModel):
    min: typing.Optional[float] = pydantic.Field(default=None)
    """
    The minimum value that can be selected.
    """

    max: typing.Optional[float] = pydantic.Field(default=None)
    """
    The maximum value that can be selected.
    """

    step: typing.Optional[float] = pydantic.Field(default=None)
    """
    The step size for the number input.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow