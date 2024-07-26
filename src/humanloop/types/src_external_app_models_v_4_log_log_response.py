# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations

import datetime as dt
import typing

import pydantic

from ..core.pydantic_utilities import IS_PYDANTIC_V2, update_forward_refs
from ..core.unchecked_base_model import UncheckedBaseModel
from .chat_message_with_tool_call import ChatMessageWithToolCall
from .config_response import ConfigResponse
from .feedback_response import FeedbackResponse
from .metric_value_response import MetricValueResponse
from .observability_status import ObservabilityStatus
from .src_external_app_models_v_4_log_log_response_judgment import SrcExternalAppModelsV4LogLogResponseJudgment
from .src_external_app_models_v_4_log_log_response_tool_choice import SrcExternalAppModelsV4LogLogResponseToolChoice
from .tool_result_response import ToolResultResponse


class SrcExternalAppModelsV4LogLogResponse(UncheckedBaseModel):
    """
    Request model for logging a datapoint.
    """

    project: typing.Optional[str] = pydantic.Field(default=None)
    """
    The name of the project associated with this log
    """

    project_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    The unique ID of the project associated with this log.
    """

    session_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    ID of the session to associate the datapoint.
    """

    session_reference_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    A unique string identifying the session to associate the datapoint to. Allows you to log multiple datapoints to a session (using an ID kept by your internal systems) by passing the same `session_reference_id` in subsequent log requests. Specify at most one of this or `session_id`.
    """

    parent_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    ID associated to the parent datapoint in a session.
    """

    parent_reference_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    A unique string identifying the previously-logged parent datapoint in a session. Allows you to log nested datapoints with your internal system IDs by passing the same reference ID as `parent_id` in a prior log request. Specify at most one of this or `parent_id`. Note that this cannot refer to a datapoint being logged in the same request.
    """

    inputs: typing.Optional[typing.Dict[str, typing.Any]] = pydantic.Field(default=None)
    """
    The inputs passed to the prompt template.
    """

    source: typing.Optional[str] = pydantic.Field(default=None)
    """
    Identifies where the model was called from.
    """

    metadata: typing.Optional[typing.Dict[str, typing.Any]] = pydantic.Field(default=None)
    """
    Any additional metadata to record.
    """

    save: typing.Optional[bool] = pydantic.Field(default=None)
    """
    Whether the request/response payloads will be stored on Humanloop.
    """

    source_datapoint_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    ID of the source datapoint if this is a log derived from a datapoint in a dataset.
    """

    id: str = pydantic.Field()
    """
    String ID of logged datapoint. Starts with `data_`.
    """

    reference_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    Unique user-provided string identifying the datapoint.
    """

    trial_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    Unique ID of an experiment trial to associate to the log.
    """

    messages: typing.Optional[typing.List[ChatMessageWithToolCall]] = pydantic.Field(default=None)
    """
    The messages passed to the to provider chat endpoint.
    """

    output: typing.Optional[str] = pydantic.Field(default=None)
    """
    Generated output from your model for the provided inputs. Can be `None` if logging an error, or if logging a parent datapoint with the intention to populate it later
    """

    judgment: typing.Optional[SrcExternalAppModelsV4LogLogResponseJudgment] = None
    config_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    Unique ID of a config to associate to the log.
    """

    config: ConfigResponse
    environment: typing.Optional[str] = pydantic.Field(default=None)
    """
    The environment name used to create the log.
    """

    feedback: typing.Optional[typing.List[FeedbackResponse]] = None
    created_at: typing.Optional[dt.datetime] = pydantic.Field(default=None)
    """
    User defined timestamp for when the log was created.
    """

    error: typing.Optional[str] = pydantic.Field(default=None)
    """
    Error message if the log is an error.
    """

    duration: typing.Optional[float] = pydantic.Field(default=None)
    """
    Duration of the logged event in seconds.
    """

    output_message: typing.Optional[ChatMessageWithToolCall] = pydantic.Field(default=None)
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

    provider_request: typing.Optional[typing.Dict[str, typing.Any]] = pydantic.Field(default=None)
    """
    Raw request sent to provider.
    """

    provider_response: typing.Optional[typing.Dict[str, typing.Any]] = pydantic.Field(default=None)
    """
    Raw response received the provider.
    """

    user: typing.Optional[str] = pydantic.Field(default=None)
    """
    User email address provided when creating the datapoint.
    """

    provider_latency: typing.Optional[float] = pydantic.Field(default=None)
    """
    Latency of provider response.
    """

    tokens: typing.Optional[int] = pydantic.Field(default=None)
    """
    Total number of tokens in the prompt and output.
    """

    raw_output: typing.Optional[str] = pydantic.Field(default=None)
    """
    Raw output from the provider.
    """

    finish_reason: typing.Optional[str] = pydantic.Field(default=None)
    """
    Reason the generation finished.
    """

    metric_values: typing.Optional[typing.List[MetricValueResponse]] = None
    tools: typing.Optional[typing.List[ToolResultResponse]] = None
    tool_choice: typing.Optional[SrcExternalAppModelsV4LogLogResponseToolChoice] = pydantic.Field(default=None)
    """
    Controls how the model uses tools. The following options are supported: 'none' forces the model to not call a tool; the default when no tools are provided as part of the model config. 'auto' the model can decide to call one of the provided tools; the default when tools are provided as part of the model config. Providing {'type': 'function', 'function': {name': <TOOL_NAME>}} forces the model to use the named function.
    """

    evaluation_results: typing.List[EvaluationResultResponse]
    observability_status: ObservabilityStatus
    updated_at: dt.datetime
    batch_ids: typing.Optional[typing.List[str]] = pydantic.Field(default=None)
    """
    List of batch IDs the log belongs to.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow


from .evaluation_result_response import EvaluationResultResponse  # noqa: E402

update_forward_refs(SrcExternalAppModelsV4LogLogResponse)