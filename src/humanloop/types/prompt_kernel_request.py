# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
import pydantic
import typing
from .model_endpoints import ModelEndpoints
from .prompt_kernel_request_template import PromptKernelRequestTemplate
from .model_providers import ModelProviders
from .prompt_kernel_request_stop import PromptKernelRequestStop
from .response_format import ResponseFormat
from .tool_function import ToolFunction
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class PromptKernelRequest(UncheckedBaseModel):
    model: str = pydantic.Field()
    """
    The model instance used, e.g. `gpt-4`. See [supported models](https://humanloop.com/docs/reference/supported-models)
    """

    endpoint: typing.Optional[ModelEndpoints] = pydantic.Field(default=None)
    """
    The provider model endpoint used.
    """

    template: typing.Optional[PromptKernelRequestTemplate] = pydantic.Field(default=None)
    """
    The template contains the main structure and instructions for the model, including input variables for dynamic values. 
    
    For chat models, provide the template as a ChatTemplate (a list of messages), e.g. a system message, followed by a user message with an input variable.
    For completion models, provide a prompt template as a string. 
    
    Input variables should be specified with double curly bracket syntax: `{{input_name}}`.
    """

    provider: typing.Optional[ModelProviders] = pydantic.Field(default=None)
    """
    The company providing the underlying model service.
    """

    max_tokens: typing.Optional[int] = pydantic.Field(default=None)
    """
    The maximum number of tokens to generate. Provide max_tokens=-1 to dynamically calculate the maximum number of tokens to generate given the length of the prompt
    """

    temperature: typing.Optional[float] = pydantic.Field(default=None)
    """
    What sampling temperature to use when making a generation. Higher values means the model will be more creative.
    """

    top_p: typing.Optional[float] = pydantic.Field(default=None)
    """
    An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
    """

    stop: typing.Optional[PromptKernelRequestStop] = pydantic.Field(default=None)
    """
    The string (or list of strings) after which the model will stop generating. The returned text will not contain the stop sequence.
    """

    presence_penalty: typing.Optional[float] = pydantic.Field(default=None)
    """
    Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the generation so far.
    """

    frequency_penalty: typing.Optional[float] = pydantic.Field(default=None)
    """
    Number between -2.0 and 2.0. Positive values penalize new tokens based on how frequently they appear in the generation so far.
    """

    other: typing.Optional[typing.Dict[str, typing.Optional[typing.Any]]] = pydantic.Field(default=None)
    """
    Other parameter values to be passed to the provider call.
    """

    seed: typing.Optional[int] = pydantic.Field(default=None)
    """
    If specified, model will make a best effort to sample deterministically, but it is not guaranteed.
    """

    response_format: typing.Optional[ResponseFormat] = pydantic.Field(default=None)
    """
    The format of the response. Only `{"type": "json_object"}` is currently supported for chat.
    """

    tools: typing.Optional[typing.List[ToolFunction]] = pydantic.Field(default=None)
    """
    The tool specification that the model can choose to call if Tool calling is supported.
    """

    linked_tools: typing.Optional[typing.List[str]] = pydantic.Field(default=None)
    """
    The IDs of the Tools in your organization that the model can choose to call if Tool calling is supported. The default deployed version of that tool is called.
    """

    attributes: typing.Optional[typing.Dict[str, typing.Optional[typing.Any]]] = pydantic.Field(default=None)
    """
    Additional fields to describe the Prompt. Helpful to separate Prompt versions from each other with details on how they were created or used.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
