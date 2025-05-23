# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel


class EvaluatorAggregate(UncheckedBaseModel):
    value: float = pydantic.Field()
    """
    The aggregated value of the evaluator.
    """

    evaluator_id: str = pydantic.Field()
    """
    ID of the evaluator.
    """

    evaluator_version_id: str = pydantic.Field()
    """
    ID of the evaluator version.
    """

    created_at: dt.datetime
    updated_at: dt.datetime

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
