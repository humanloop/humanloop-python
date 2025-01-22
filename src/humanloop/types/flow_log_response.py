# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations
from ..core.unchecked_base_model import UncheckedBaseModel
from .evaluator_response import EvaluatorResponse
from .flow_response import FlowResponse
from .monitoring_evaluator_response import MonitoringEvaluatorResponse
from .prompt_response import PromptResponse
from .tool_response import ToolResponse
from .version_deployment_response import VersionDeploymentResponse
from .version_id_response import VersionIdResponse
import typing
from .chat_message import ChatMessage
import pydantic
import datetime as dt
from .trace_status import TraceStatus
from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.pydantic_utilities import update_forward_refs


class FlowLogResponse(UncheckedBaseModel):
    """
    General request for creating a Log
    """

    messages: typing.Optional[typing.List[ChatMessage]] = pydantic.Field(default=None)
    """
    List of chat messages that were used as an input to the Flow.
    """

    output_message: typing.Optional[ChatMessage] = pydantic.Field(default=None)
    """
    The output message returned by this Flow.
    """

    start_time: typing.Optional[dt.datetime] = pydantic.Field(default=None)
    """
    When the logged event started.
    """

    end_time: typing.Optional[dt.datetime] = pydantic.Field(default=None)
    """
    When the logged event ended.
    """

    output: typing.Optional[str] = pydantic.Field(default=None)
    """
    Generated output from your model for the provided inputs. Can be `None` if logging an error, or if creating a parent Log with the intention to populate it later.
    """

    created_at: typing.Optional[dt.datetime] = pydantic.Field(default=None)
    """
    User defined timestamp for when the log was created. 
    """

    error: typing.Optional[str] = pydantic.Field(default=None)
    """
    Error message if the log is an error.
    """

    provider_latency: typing.Optional[float] = pydantic.Field(default=None)
    """
    Duration of the logged event in seconds.
    """

    stdout: typing.Optional[str] = pydantic.Field(default=None)
    """
    Captured log and debug statements.
    """

    provider_request: typing.Optional[typing.Dict[str, typing.Optional[typing.Any]]] = pydantic.Field(default=None)
    """
    Raw request sent to provider.
    """

    provider_response: typing.Optional[typing.Dict[str, typing.Optional[typing.Any]]] = pydantic.Field(default=None)
    """
    Raw response received the provider.
    """

    inputs: typing.Optional[typing.Dict[str, typing.Optional[typing.Any]]] = pydantic.Field(default=None)
    """
    The inputs passed to the Flow Log.
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

    batches: typing.Optional[typing.List[str]] = pydantic.Field(default=None)
    """
    Array of Batch IDs that this Log is part of. Batches are used to group Logs together for offline Evaluations
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
    Unique identifier for the Log.
    """

    evaluator_logs: typing.List["EvaluatorLogResponse"] = pydantic.Field()
    """
    List of Evaluator Logs associated with the Log. These contain Evaluator judgments on the Log.
    """

    trace_flow_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    Identifier for the Flow that the Trace belongs to.
    """

    trace_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    Identifier for the Trace that the Log belongs to.
    """

    trace_children: typing.Optional[typing.List["LogResponse"]] = pydantic.Field(default=None)
    """
    Logs nested under this Log in the Trace.
    """

    flow: FlowResponse = pydantic.Field()
    """
    Flow used to generate the Log.
    """

    trace_status: typing.Optional[TraceStatus] = pydantic.Field(default=None)
    """
    Status of the Trace. When a Trace is marked as `complete`, no more Logs can be added to it. Monitoring Evaluators will only run on completed Traces.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow


from .evaluator_log_response import EvaluatorLogResponse  # noqa: E402
from .prompt_log_response import PromptLogResponse  # noqa: E402
from .tool_log_response import ToolLogResponse  # noqa: E402
from .log_response import LogResponse  # noqa: E402

update_forward_refs(EvaluatorLogResponse, FlowLogResponse=FlowLogResponse)
update_forward_refs(EvaluatorResponse, FlowLogResponse=FlowLogResponse)
update_forward_refs(FlowResponse, FlowLogResponse=FlowLogResponse)
update_forward_refs(MonitoringEvaluatorResponse, FlowLogResponse=FlowLogResponse)
update_forward_refs(PromptLogResponse, FlowLogResponse=FlowLogResponse)
update_forward_refs(PromptResponse, FlowLogResponse=FlowLogResponse)
update_forward_refs(ToolLogResponse, FlowLogResponse=FlowLogResponse)
update_forward_refs(ToolResponse, FlowLogResponse=FlowLogResponse)
update_forward_refs(VersionDeploymentResponse, FlowLogResponse=FlowLogResponse)
update_forward_refs(VersionIdResponse, FlowLogResponse=FlowLogResponse)
update_forward_refs(FlowLogResponse)
