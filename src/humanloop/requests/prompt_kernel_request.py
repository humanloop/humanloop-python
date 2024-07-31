# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions

from ..types.model_endpoints import ModelEndpoints
from ..types.model_providers import ModelProviders
from .prompt_kernel_request_stop import PromptKernelRequestStopParams
from .prompt_kernel_request_template import PromptKernelRequestTemplateParams
from .response_format import ResponseFormatParams
from .tool_function import ToolFunctionParams


class PromptKernelRequestParams(typing_extensions.TypedDict):
    model: str
    """
    The model instance used, e.g. `gpt-4`. See [supported models](https://humanloop.com/docs/supported-models)
    """

    endpoint: typing_extensions.NotRequired[ModelEndpoints]
    """
    The provider model endpoint used.
    """

    template: typing_extensions.NotRequired[PromptKernelRequestTemplateParams]
    """
    For chat endpoint, provide a Chat template. For completion endpoint, provide a Prompt template. Input variables within the template should be specified with double curly bracket syntax: {{INPUT_NAME}}.
    """

    provider: typing_extensions.NotRequired[ModelProviders]
    """
    The company providing the underlying model service.
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

    stop: typing_extensions.NotRequired[PromptKernelRequestStopParams]
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
    The format of the response. Only `{"type": "json_object"}` is currently supported for chat.
    """

    tools: typing_extensions.NotRequired[typing.Sequence[ToolFunctionParams]]
    """
    The tool specification that the model can choose to call if Tool calling is supported.
    """

    linked_tools: typing_extensions.NotRequired[typing.Sequence[str]]
    """
    The IDs of the Tools in your organization that the model can choose to call if Tool calling is supported. The default deployed version of that tool is called.
    """
