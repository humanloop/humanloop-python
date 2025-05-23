# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations

import datetime as dt
import typing

import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2, update_forward_refs
from ..core.unchecked_base_model import UncheckedBaseModel
from .chat_message import ChatMessage
from .prompt_call_log_response import PromptCallLogResponse
from .prompt_call_response_tool_choice import PromptCallResponseToolChoice


class PromptCallResponse(UncheckedBaseModel):
    """
    Response model for a Prompt call with potentially multiple log samples.
    """

    start_time: typing.Optional[dt.datetime] = pydantic.Field(default=None)
    """
    When the logged event started.
    """

    end_time: typing.Optional[dt.datetime] = pydantic.Field(default=None)
    """
    When the logged event ended.
    """

    messages: typing.Optional[typing.List[ChatMessage]] = pydantic.Field(default=None)
    """
    The messages passed to the to provider chat endpoint.
    """

    tool_choice: typing.Optional[PromptCallResponseToolChoice] = pydantic.Field(default=None)
    """
    Controls how the model uses tools. The following options are supported: 
    - `'none'` means the model will not call any tool and instead generates a message; this is the default when no tools are provided as part of the Prompt. 
    - `'auto'` means the model can decide to call one or more of the provided tools; this is the default when tools are provided as part of the Prompt. 
    - `'required'` means the model must call one or more of the provided tools. 
    - `{'type': 'function', 'function': {name': <TOOL_NAME>}}` forces the model to use the named function.
    """

    prompt: "PromptResponse" = pydantic.Field()
    """
    Prompt used to generate the Log.
    """

    inputs: typing.Optional[typing.Dict[str, typing.Optional[typing.Any]]] = pydantic.Field(default=None)
    """
    The inputs passed to the prompt template.
    """

    source: typing.Optional[str] = pydantic.Field(default=None)
    """
    Identifies where the model was called from.
    """

    metadata: typing.Optional[typing.Dict[str, typing.Optional[typing.Any]]] = pydantic.Field(default=None)
    """
    Any additional metadata to record.
    """

    source_datapoint_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    Unique identifier for the Datapoint that this Log is derived from. This can be used by Humanloop to associate Logs to Evaluations. If provided, Humanloop will automatically associate this Log to Evaluations that require a Log for this Datapoint-Version pair.
    """

    trace_parent_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    The ID of the parent Log to nest this Log under in a Trace.
    """

    user: typing.Optional[str] = pydantic.Field(default=None)
    """
    End-user ID related to the Log.
    """

    environment: typing.Optional[str] = pydantic.Field(default=None)
    """
    The name of the Environment the Log is associated to.
    """

    save: typing.Optional[bool] = pydantic.Field(default=None)
    """
    Whether the request/response payloads will be stored on Humanloop.
    """

    log_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    This will identify a Log. If you don't provide a Log ID, Humanloop will generate one for you.
    """

    id: str = pydantic.Field()
    """
    ID of the log.
    """

    trace_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    ID of the Trace containing the Prompt Call Log.
    """

    logs: typing.List[PromptCallLogResponse] = pydantic.Field()
    """
    The logs generated by the Prompt call.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow


from .agent_linked_file_response import AgentLinkedFileResponse  # noqa: E402, F401, I001
from .agent_response import AgentResponse  # noqa: E402, F401, I001
from .evaluator_response import EvaluatorResponse  # noqa: E402, F401, I001
from .flow_response import FlowResponse  # noqa: E402, F401, I001
from .monitoring_evaluator_response import MonitoringEvaluatorResponse  # noqa: E402, F401, I001
from .prompt_response import PromptResponse  # noqa: E402, F401, I001
from .tool_response import ToolResponse  # noqa: E402, F401, I001
from .version_deployment_response import VersionDeploymentResponse  # noqa: E402, F401, I001
from .version_id_response import VersionIdResponse  # noqa: E402, F401, I001

update_forward_refs(PromptCallResponse)
