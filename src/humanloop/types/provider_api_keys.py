# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic

from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel


class ProviderApiKeys(UncheckedBaseModel):
    openai: typing.Optional[str] = None
    ai_21: typing.Optional[str] = pydantic.Field(alias="ai21", default=None)
    mock: typing.Optional[str] = None
    anthropic: typing.Optional[str] = None
    cohere: typing.Optional[str] = None
    openai_azure: typing.Optional[str] = None
    openai_azure_endpoint: typing.Optional[str] = None

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
