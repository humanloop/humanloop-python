# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations
import typing_extensions
import typing_extensions
from ..types.model_endpoints import ModelEndpoints
from .prompt_response_template import PromptResponseTemplateParams
from ..types.model_providers import ModelProviders
from .prompt_response_stop import PromptResponseStopParams
import typing
from .response_format import ResponseFormatParams
from .tool_function import ToolFunctionParams
from .linked_tool_response import LinkedToolResponseParams
from .environment_response import EnvironmentResponseParams
import datetime as dt
from ..types.user_response import UserResponse
from ..types.version_status import VersionStatus
from .input_response import InputResponseParams
from .evaluator_aggregate import EvaluatorAggregateParams
import typing

if typing.TYPE_CHECKING:
    from .monitoring_evaluator_response import MonitoringEvaluatorResponseParams


class PromptResponseParams(typing_extensions.TypedDict):
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
    The model instance used, e.g. `gpt-4`. See [supported models](https://humanloop.com/docs/supported-models)
    """

    endpoint: typing_extensions.NotRequired[ModelEndpoints]
    """
    The provider model endpoint used.
    """

    template: typing_extensions.NotRequired[PromptResponseTemplateParams]
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

    stop: typing_extensions.NotRequired[PromptResponseStopParams]
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

    tools: typing_extensions.NotRequired[typing.Sequence[ToolFunctionParams]]
    """
    The tool specification that the model can choose to call if Tool calling is supported.
    """

    linked_tools: typing_extensions.NotRequired[typing.Sequence[LinkedToolResponseParams]]
    """
    The tools linked to your prompt that the model can call.
    """

    commit_message: typing_extensions.NotRequired[str]
    """
    Message describing the changes made.
    """

    name: str
    """
    Name of the Prompt.
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

    status: VersionStatus
    """
    The status of the Prompt Version.
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

    evaluators: typing_extensions.NotRequired[typing.Sequence["MonitoringEvaluatorResponseParams"]]
    """
    Evaluators that have been attached to this Prompt that are used for monitoring logs.
    """

    evaluator_aggregates: typing_extensions.NotRequired[typing.Sequence[EvaluatorAggregateParams]]
    """
    Aggregation of Evaluator results for the Prompt Version.
    """
