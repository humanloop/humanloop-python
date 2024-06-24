# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ..core.datetime_utils import serialize_datetime
from ..core.pydantic_utilities import deep_union_pydantic_dicts, pydantic_v1
from ..core.unchecked_base_model import UncheckedBaseModel
from .version_stats_evaluator_version_stats_item import VersionStatsEvaluatorVersionStatsItem


class VersionStats(UncheckedBaseModel):
    """
    Stats for an Evaluated Version in the Evaluation Report.
    """

    version_id: str = pydantic_v1.Field()
    """
    Unique identifier for the Evaluated Version.
    """

    num_logs: int = pydantic_v1.Field()
    """
    The total number of existing Logs for this Evaluated Version within the Evaluation Report. These are Logs that have been generated by this Evaluated Version on a Datapoint belonging to the Evaluation Report's Dataset Version.
    """

    evaluator_version_stats: typing.List[VersionStatsEvaluatorVersionStatsItem] = pydantic_v1.Field()
    """
    Stats for each Evaluator Version used to evaluate this Evaluated Version.
    """

    def json(self, **kwargs: typing.Any) -> str:
        kwargs_with_defaults: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        return super().json(**kwargs_with_defaults)

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        kwargs_with_defaults_exclude_unset: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        kwargs_with_defaults_exclude_none: typing.Any = {"by_alias": True, "exclude_none": True, **kwargs}

        return deep_union_pydantic_dicts(
            super().dict(**kwargs_with_defaults_exclude_unset), super().dict(**kwargs_with_defaults_exclude_none)
        )

    class Config:
        frozen = True
        smart_union = True
        extra = pydantic_v1.Extra.allow
        json_encoders = {dt.datetime: serialize_datetime}
