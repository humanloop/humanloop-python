# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions

from .categorical_feedback_label import CategoricalFeedbackLabelParams
from .feedback_type_model_type import FeedbackTypeModelTypeParams


class FeedbackTypeModelParams(typing_extensions.TypedDict):
    type: FeedbackTypeModelTypeParams
    """
    The type of feedback. The default feedback types available are 'rating', 'action', 'issue', 'correction', and 'comment'.
    """

    values: typing_extensions.NotRequired[typing.Sequence[CategoricalFeedbackLabelParams]]
    """
    The allowed values for categorical feedback types. Not populated for `correction` and `comment`.
    """