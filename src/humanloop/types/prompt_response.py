# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations
from ..core.unchecked_base_model import UncheckedBaseModel
import pydantic
import typing
from .model_endpoints import ModelEndpoints
from .prompt_response_template import PromptResponseTemplate
from .template_language import TemplateLanguage
from .model_providers import ModelProviders
from .prompt_response_stop import PromptResponseStop
from .response_format import ResponseFormat
from .prompt_response_reasoning_effort import PromptResponseReasoningEffort
from .tool_function import ToolFunction
from .linked_tool_response import LinkedToolResponse
import typing_extensions
from ..core.serialization import FieldMetadata
from .environment_response import EnvironmentResponse
import datetime as dt
from .user_response import UserResponse
from .input_response import InputResponse
from .evaluator_aggregate import EvaluatorAggregate
from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.pydantic_utilities import update_forward_refs


class PromptResponse(UncheckedBaseModel):
    """
    Base type that all File Responses should inherit from.

    Attributes defined here are common to all File Responses and should be overridden
    in the inheriting classes with documentation and appropriate Field definitions.
    """

    path: str = pydantic.Field()
    """
    Path of the Prompt, including the name, which is used as a unique identifier.
    """

    id: str = pydantic.Field()
    """
    Unique identifier for the Prompt.
    """

    directory_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    ID of the directory that the file is in on Humanloop.
    """

    model: str = pydantic.Field()
    """
    The model instance used, e.g. `gpt-4`. See [supported models](https://humanloop.com/docs/reference/supported-models)
    """

    endpoint: typing.Optional[ModelEndpoints] = pydantic.Field(default=None)
    """
    The provider model endpoint used.
    """

    template: typing.Optional[PromptResponseTemplate] = pydantic.Field(default=None)
    """
    The template contains the main structure and instructions for the model, including input variables for dynamic values. 
    
    For chat models, provide the template as a ChatTemplate (a list of messages), e.g. a system message, followed by a user message with an input variable.
    For completion models, provide a prompt template as a string. 
    
    Input variables should be specified with double curly bracket syntax: `{{input_name}}`.
    """

    template_language: typing.Optional[TemplateLanguage] = pydantic.Field(default=None)
    """
    The template language to use for rendering the template.
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

    stop: typing.Optional[PromptResponseStop] = pydantic.Field(default=None)
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

    reasoning_effort: typing.Optional[PromptResponseReasoningEffort] = pydantic.Field(default=None)
    """
    Guidance on how many reasoning tokens it should generate before creating a response to the prompt. OpenAI reasoning models (o1, o3-mini) expect a OpenAIReasoningEffort enum. Anthropic reasoning models expect an integer, which signifies the maximum token budget.
    """

    tools: typing.Optional[typing.List[ToolFunction]] = pydantic.Field(default=None)
    """
    The tool specification that the model can choose to call if Tool calling is supported.
    """

    linked_tools: typing.Optional[typing.List[LinkedToolResponse]] = pydantic.Field(default=None)
    """
    The tools linked to your prompt that the model can call.
    """

    attributes: typing.Optional[typing.Dict[str, typing.Optional[typing.Any]]] = pydantic.Field(default=None)
    """
    Additional fields to describe the Prompt. Helpful to separate Prompt versions from each other with details on how they were created or used.
    """

    version_name: typing.Optional[str] = pydantic.Field(default=None)
    """
    Unique name for the Prompt version. Version names must be unique for a given Prompt.
    """

    version_description: typing.Optional[str] = pydantic.Field(default=None)
    """
    Description of the version, e.g., the changes made in this version.
    """

    description: typing.Optional[str] = pydantic.Field(default=None)
    """
    Description of the Prompt.
    """

    tags: typing.Optional[typing.List[str]] = pydantic.Field(default=None)
    """
    List of tags associated with the file.
    """

    readme: typing.Optional[str] = pydantic.Field(default=None)
    """
    Long description of the file.
    """

    name: str = pydantic.Field()
    """
    Name of the Prompt.
    """

    schema_: typing_extensions.Annotated[
        typing.Optional[typing.Dict[str, typing.Optional[typing.Any]]], FieldMetadata(alias="schema")
    ] = pydantic.Field(default=None)
    """
    The JSON schema for the Prompt.
    """

    version_id: str = pydantic.Field()
    """
    Unique identifier for the specific Prompt Version. If no query params provided, the default deployed Prompt Version is returned.
    """

    type: typing.Optional[typing.Literal["prompt"]] = None
    environments: typing.Optional[typing.List[EnvironmentResponse]] = pydantic.Field(default=None)
    """
    The list of environments the Prompt Version is deployed to.
    """

    created_at: dt.datetime
    updated_at: dt.datetime
    created_by: typing.Optional[UserResponse] = pydantic.Field(default=None)
    """
    The user who created the Prompt.
    """

    last_used_at: dt.datetime
    version_logs_count: int = pydantic.Field()
    """
    The number of logs that have been generated for this Prompt Version
    """

    total_logs_count: int = pydantic.Field()
    """
    The number of logs that have been generated across all Prompt Versions
    """

    inputs: typing.List[InputResponse] = pydantic.Field()
    """
    Inputs associated to the Prompt. Inputs correspond to any of the variables used within the Prompt template.
    """

    evaluators: typing.Optional[typing.List["MonitoringEvaluatorResponse"]] = pydantic.Field(default=None)
    """
    Evaluators that have been attached to this Prompt that are used for monitoring logs.
    """

    evaluator_aggregates: typing.Optional[typing.List[EvaluatorAggregate]] = pydantic.Field(default=None)
    """
    Aggregation of Evaluator results for the Prompt Version.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow


from .agent_linked_file_response import AgentLinkedFileResponse  # noqa: E402
from .agent_response import AgentResponse  # noqa: E402
from .evaluator_response import EvaluatorResponse  # noqa: E402
from .flow_response import FlowResponse  # noqa: E402
from .monitoring_evaluator_response import MonitoringEvaluatorResponse  # noqa: E402
from .tool_response import ToolResponse  # noqa: E402
from .version_deployment_response import VersionDeploymentResponse  # noqa: E402
from .version_id_response import VersionIdResponse  # noqa: E402

update_forward_refs(PromptResponse)
