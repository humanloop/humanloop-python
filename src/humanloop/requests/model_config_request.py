# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions

from ..types.model_endpoints import ModelEndpoints
from ..types.model_providers import ModelProviders
from .chat_message_with_tool_call import ChatMessageWithToolCallParams
from .model_config_request_stop import ModelConfigRequestStopParams
from .model_config_request_tools_item import ModelConfigRequestToolsItemParams
from .response_format import ResponseFormatParams


class ModelConfigRequestParams(typing_extensions.TypedDict):
    """
    Model config used for logging both chat and completion.
    """

    name: typing_extensions.NotRequired[str]
    """
    A friendly display name for the model config. If not provided, a name will be generated.
    """

    description: typing_extensions.NotRequired[str]
    """
    A description of the model config.
    """

    provider: typing_extensions.NotRequired[ModelProviders]
    """
    The company providing the underlying model service.
    """

    model: str
    """
    The model instance used. E.g. text-davinci-002.
    """

    max_tokens: typing_extensions.NotRequired[int]
    """
    The maximum number of tokens to generate. Provide max_tokens=-1 to dynamically calculate the maximum number of tokens to generate given the length of the prompt
    """

    temperature: typing_extensions.NotRequired[float]
    """
    What sampling temperature to use when making a generation. Higher values means the model will be more creative.
    """

    top_p: typing_extensions.NotRequired[float]
    """
    An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
    """

    stop: typing_extensions.NotRequired[ModelConfigRequestStopParams]
    """
    The string (or list of strings) after which the model will stop generating. The returned text will not contain the stop sequence.
    """

    presence_penalty: typing_extensions.NotRequired[float]
    """
    Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the generation so far.
    """

    frequency_penalty: typing_extensions.NotRequired[float]
    """
    Number between -2.0 and 2.0. Positive values penalize new tokens based on how frequently they appear in the generation so far.
    """

    other: typing_extensions.NotRequired[typing.Dict[str, typing.Any]]
    """
    Other parameter values to be passed to the provider call.
    """

    seed: typing_extensions.NotRequired[int]
    """
    If specified, model will make a best effort to sample deterministically, but it is not guaranteed.
    """

    response_format: typing_extensions.NotRequired[ResponseFormatParams]
    """
    The format of the response. Only type json_object is currently supported for chat.
    """

    endpoint: typing_extensions.NotRequired[ModelEndpoints]
    """
    The provider model endpoint used.
    """

    prompt_template: typing_extensions.NotRequired[str]
    """
    Prompt template that will take your specified inputs to form your final request to the model. Input variables within the prompt template should be specified with syntax: {{INPUT_NAME}}.
    """

    chat_template: typing_extensions.NotRequired[typing.Sequence[ChatMessageWithToolCallParams]]
    """
    Messages prepended to the list of messages sent to the provider. These messages that will take your specified inputs to form your final request to the provider model. Input variables within the template should be specified with syntax: {{INPUT_NAME}}.
    """

    tools: typing_extensions.NotRequired[typing.Sequence[ModelConfigRequestToolsItemParams]]
    """
    Make tools available to OpenAIs chat model as functions.
    """

    type: typing_extensions.NotRequired[typing.Literal["model"]]