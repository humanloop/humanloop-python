# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ..core.datetime_utils import serialize_datetime
from ..core.pydantic_utilities import deep_union_pydantic_dicts, pydantic_v1
from ..core.unchecked_base_model import UncheckedBaseModel
from .evaluated_version_response import EvaluatedVersionResponse


class EvaluateeResponse(UncheckedBaseModel):
    """
    Version of the Evaluatee being evaluated.
    """

    version: EvaluatedVersionResponse
    batch_id: typing.Optional[str] = pydantic_v1.Field(default=None)
    """
    Unique identifier for the batch of Logs to include in the Evaluation Report.
    """

    orchestrated: bool = pydantic_v1.Field()
    """
    Whether the Prompt/Tool is orchestrated by Humanloop. Default is `True`. If `False`, a log for the Prompt/Tool should be submitted by the user via the API.
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
