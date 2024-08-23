# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import typing_extensions
from ..types.valence import Valence


class EvaluatorJudgmentOptionResponseParams(typing_extensions.TypedDict):
    name: str
    """
    The name of the option.
    """

    valence: typing_extensions.NotRequired[Valence]
    """
    Whether this option should be considered positive or negative.
    """