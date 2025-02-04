# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import typing_extensions
from ..core.serialization import FieldMetadata


class ProviderApiKeysParams(typing_extensions.TypedDict):
    openai: typing_extensions.NotRequired[str]
    ai_21: typing_extensions.NotRequired[typing_extensions.Annotated[str, FieldMetadata(alias="ai21")]]
    mock: typing_extensions.NotRequired[str]
    anthropic: typing_extensions.NotRequired[str]
    deepseek: typing_extensions.NotRequired[str]
    bedrock: typing_extensions.NotRequired[str]
    cohere: typing_extensions.NotRequired[str]
    openai_azure: typing_extensions.NotRequired[str]
    openai_azure_endpoint: typing_extensions.NotRequired[str]
