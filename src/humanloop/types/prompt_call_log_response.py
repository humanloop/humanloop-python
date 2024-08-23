# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

import pydantic

from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel
from .chat_message import ChatMessage


class PromptCallLogResponse(UncheckedBaseModel):
    """
    Sample specific response details for a Prompt call
    """

    output: typing.Optional[str] = pydantic.Field(default=None)
    """
    Generated output from your model for the provided inputs. Can be `None` if logging an error, or if creating a parent Log with the intention to populate it later.
    """

    created_at: typing.Optional[dt.datetime] = pydantic.Field(default=None)
    """
    User defined timestamp for when the log was created.
    """

    error: typing.Optional[str] = pydantic.Field(default=None)
    """
    Error message if the log is an error.
    """

    provider_latency: typing.Optional[float] = pydantic.Field(default=None)
    """
    Duration of the logged event in seconds.
    """

    output_message: typing.Optional[ChatMessage] = pydantic.Field(default=None)
    """
    The message returned by the provider.
    """

    prompt_tokens: typing.Optional[int] = pydantic.Field(default=None)
    """
    Number of tokens in the prompt used to generate the output.
    """

    output_tokens: typing.Optional[int] = pydantic.Field(default=None)
    """
    Number of tokens in the output generated by the model.
    """

    prompt_cost: typing.Optional[float] = pydantic.Field(default=None)
    """
    Cost in dollars associated to the tokens in the prompt.
    """

    output_cost: typing.Optional[float] = pydantic.Field(default=None)
    """
    Cost in dollars associated to the tokens in the output.
    """

    finish_reason: typing.Optional[str] = pydantic.Field(default=None)
    """
    Reason the generation finished.
    """

    index: int = pydantic.Field()
    """
    The index of the sample in the batch.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
