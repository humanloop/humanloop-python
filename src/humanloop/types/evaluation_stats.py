# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
import typing
from .run_stats_response import RunStatsResponse
import pydantic
from .evaluation_status import EvaluationStatus
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class EvaluationStats(UncheckedBaseModel):
    run_stats: typing.List[RunStatsResponse] = pydantic.Field()
    """
    Stats for each Run in the Evaluation.
    """

    progress: typing.Optional[str] = pydantic.Field(default=None)
    """
    A summary string report of the Evaluation's progress you can print to the command line;helpful when integrating Evaluations with CI/CD.
    """

    report: typing.Optional[str] = pydantic.Field(default=None)
    """
    A summary string report of the Evaluation you can print to command line;helpful when integrating Evaluations with CI/CD.
    """

    status: EvaluationStatus = pydantic.Field()
    """
    The current status of the Evaluation.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
