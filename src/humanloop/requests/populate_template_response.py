# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

import typing_extensions
from ..types.model_endpoints import ModelEndpoints
from ..types.model_providers import ModelProviders
from ..types.template_language import TemplateLanguage
from ..types.user_response import UserResponse
from .environment_response import EnvironmentResponseParams
from .evaluator_aggregate import EvaluatorAggregateParams
from .input_response import InputResponseParams
from .linked_tool_response import LinkedToolResponseParams
from .monitoring_evaluator_response import MonitoringEvaluatorResponseParams
from .populate_template_response_populated_template import PopulateTemplateResponsePopulatedTemplateParams
from .populate_template_response_reasoning_effort import PopulateTemplateResponseReasoningEffortParams
from .populate_template_response_stop import PopulateTemplateResponseStopParams
from .populate_template_response_template import PopulateTemplateResponseTemplateParams
from .response_format import ResponseFormatParams
from .tool_function import ToolFunctionParams


class PopulateTemplateResponseParams(typing_extensions.TypedDict):
    """
    Base type that all File Responses should inherit from.

    Attributes defined here are common to all File Responses and should be overridden
    in the inheriting classes with documentation and appropriate Field definitions.
    """

    path: str
    """
    Path of the Prompt, including the name, which is used as a unique identifier.
    """

    id: str
    """
    Unique identifier for the Prompt.
    """

    directory_id: typing_extensions.NotRequired[str]
    """
    ID of the directory that the file is in on Humanloop.
    """

    model: str
    """
    The model instance used, e.g. `gpt-4`. See [supported models](https://humanloop.com/docs/reference/supported-models)
    """

    endpoint: typing_extensions.NotRequired[ModelEndpoints]
    """
    The provider model endpoint used.
    """

    template: typing_extensions.NotRequired[PopulateTemplateResponseTemplateParams]
    """
    The template contains the main structure and instructions for the model, including input variables for dynamic values. 
    
    For chat models, provide the template as a ChatTemplate (a list of messages), e.g. a system message, followed by a user message with an input variable.
    For completion models, provide a prompt template as a string. 
    
    Input variables should be specified with double curly bracket syntax: `{{input_name}}`.
    """

    template_language: typing_extensions.NotRequired[TemplateLanguage]
    """
    The template language to use for rendering the template.
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

    stop: typing_extensions.NotRequired[PopulateTemplateResponseStopParams]
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

    other: typing_extensions.NotRequired[typing.Dict[str, typing.Optional[typing.Any]]]
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

    reasoning_effort: typing_extensions.NotRequired[PopulateTemplateResponseReasoningEffortParams]
    """
    Guidance on how many reasoning tokens it should generate before creating a response to the prompt. OpenAI reasoning models (o1, o3-mini) expect a OpenAIReasoningEffort enum. Anthropic reasoning models expect an integer, which signifies the maximum token budget.
    """

    tools: typing_extensions.NotRequired[typing.Sequence[ToolFunctionParams]]
    """
    The tool specification that the model can choose to call if Tool calling is supported.
    """

    linked_tools: typing_extensions.NotRequired[typing.Sequence[LinkedToolResponseParams]]
    """
    The tools linked to your prompt that the model can call.
    """

    attributes: typing_extensions.NotRequired[typing.Dict[str, typing.Optional[typing.Any]]]
    """
    Additional fields to describe the Prompt. Helpful to separate Prompt versions from each other with details on how they were created or used.
    """

    version_name: typing_extensions.NotRequired[str]
    """
    Unique name for the Prompt version. Version names must be unique for a given Prompt.
    """

    version_description: typing_extensions.NotRequired[str]
    """
    Description of the version, e.g., the changes made in this version.
    """

    description: typing_extensions.NotRequired[str]
    """
    Description of the Prompt.
    """

    tags: typing_extensions.NotRequired[typing.Sequence[str]]
    """
    List of tags associated with the file.
    """

    readme: typing_extensions.NotRequired[str]
    """
    Long description of the file.
    """

    name: str
    """
    Name of the Prompt.
    """

    schema: typing_extensions.NotRequired[typing.Dict[str, typing.Optional[typing.Any]]]
    """
    The JSON schema for the Prompt.
    """

    version_id: str
    """
    Unique identifier for the specific Prompt Version. If no query params provided, the default deployed Prompt Version is returned.
    """

    type: typing_extensions.NotRequired[typing.Literal["prompt"]]
    environments: typing_extensions.NotRequired[typing.Sequence[EnvironmentResponseParams]]
    """
    The list of environments the Prompt Version is deployed to.
    """

    created_at: dt.datetime
    updated_at: dt.datetime
    created_by: typing_extensions.NotRequired[UserResponse]
    """
    The user who created the Prompt.
    """

    last_used_at: dt.datetime
    version_logs_count: int
    """
    The number of logs that have been generated for this Prompt Version
    """

    total_logs_count: int
    """
    The number of logs that have been generated across all Prompt Versions
    """

    inputs: typing.Sequence[InputResponseParams]
    """
    Inputs associated to the Prompt. Inputs correspond to any of the variables used within the Prompt template.
    """

    evaluators: typing_extensions.NotRequired[typing.Sequence[MonitoringEvaluatorResponseParams]]
    """
    Evaluators that have been attached to this Prompt that are used for monitoring logs.
    """

    evaluator_aggregates: typing_extensions.NotRequired[typing.Sequence[EvaluatorAggregateParams]]
    """
    Aggregation of Evaluator results for the Prompt Version.
    """

    raw_file_content: typing_extensions.NotRequired[str]
    """
    The raw content of the Prompt. Corresponds to the .prompt file.
    """

    populated_template: typing_extensions.NotRequired[PopulateTemplateResponsePopulatedTemplateParams]
    """
    The template populated with the input values you provided in the request. Returns None if no template exists.
    """
