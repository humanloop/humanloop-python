# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic

from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel
from .categorical_feedback_label import CategoricalFeedbackLabel
from .feedback_type_model_type import FeedbackTypeModelType


class FeedbackTypeModel(UncheckedBaseModel):
    type: FeedbackTypeModelType = pydantic.Field()
    """
    The type of feedback. The default feedback types available are 'rating', 'action', 'issue', 'correction', and 'comment'.
    """

    values: typing.Optional[typing.List[CategoricalFeedbackLabel]] = pydantic.Field(default=None)
    """
    The allowed values for categorical feedback types. Not populated for `correction` and `comment`.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
