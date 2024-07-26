# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic

from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel


class BooleanEvaluatorVersionStats(UncheckedBaseModel):
    """
    Base attributes for stats for an Evaluator Version-Evaluated Version pair
    in the Evaluation Report.
    """

    evaluator_version_id: str = pydantic.Field()
    """
    Unique identifier for the Evaluator Version.
    """

    total_logs: int = pydantic.Field()
    """
    The total number of Logs generated by this Evaluator Version on the Evaluated Version's Logs. This includes Nulls and Errors.
    """

    num_judgments: int = pydantic.Field()
    """
    The total number of Evaluator judgments for this Evaluator Version. This excludes Nulls and Errors.
    """

    num_nulls: int = pydantic.Field()
    """
    The total number of null judgments (i.e. abstentions) for this Evaluator Version.
    """

    num_errors: int = pydantic.Field()
    """
    The total number of errored Evaluators for this Evaluator Version.
    """

    num_true: int = pydantic.Field()
    """
    The total number of `True` judgments for this Evaluator Version.
    """

    num_false: int = pydantic.Field()
    """
    The total number of `False` judgments for this Evaluator Version.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
