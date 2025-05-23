# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel


class AnthropicThinkingContent(UncheckedBaseModel):
    type: typing.Literal["thinking"] = "thinking"
    thinking: str = pydantic.Field()
    """
    Model's chain-of-thought for providing the response.
    """

    signature: str = pydantic.Field()
    """
    Cryptographic signature that verifies the thinking block was generated by Anthropic.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
