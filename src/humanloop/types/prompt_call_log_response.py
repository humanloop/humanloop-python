# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ..core.datetime_utils import serialize_datetime
from ..core.pydantic_utilities import deep_union_pydantic_dicts, pydantic_v1
from ..core.unchecked_base_model import UncheckedBaseModel
from .chat_message import ChatMessage


class PromptCallLogResponse(UncheckedBaseModel):
    """
    Sample specific response details for a Prompt call
    """

    output: typing.Optional[str] = pydantic_v1.Field(default=None)
    """
    Generated output from your model for the provided inputs. Can be `None` if logging an error, or if creating a parent Log with the intention to populate it later.
    """

    raw_output: typing.Optional[str] = pydantic_v1.Field(default=None)
    """
    Raw output from the provider.
    """

    created_at: typing.Optional[dt.datetime] = pydantic_v1.Field(default=None)
    """
    User defined timestamp for when the log was created.
    """

    error: typing.Optional[str] = pydantic_v1.Field(default=None)
    """
    Error message if the log is an error.
    """

    provider_latency: typing.Optional[float] = pydantic_v1.Field(default=None)
    """
    Duration of the logged event in seconds.
    """

    output_message: typing.Optional[ChatMessage] = pydantic_v1.Field(default=None)
    """
    The message returned by the provider.
    """

    prompt_tokens: typing.Optional[int] = pydantic_v1.Field(default=None)
    """
    Number of tokens in the prompt used to generate the output.
    """

    output_tokens: typing.Optional[int] = pydantic_v1.Field(default=None)
    """
    Number of tokens in the output generated by the model.
    """

    prompt_cost: typing.Optional[float] = pydantic_v1.Field(default=None)
    """
    Cost in dollars associated to the tokens in the prompt.
    """

    output_cost: typing.Optional[float] = pydantic_v1.Field(default=None)
    """
    Cost in dollars associated to the tokens in the output.
    """

    finish_reason: typing.Optional[str] = pydantic_v1.Field(default=None)
    """
    Reason the generation finished.
    """

    index: int = pydantic_v1.Field()
    """
    The index of the sample in the batch.
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
