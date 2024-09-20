# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations
import typing_extensions
import typing_extensions
import datetime as dt
import typing
from .flow_response import FlowResponseParams
from ..types.trace_status import TraceStatus
import typing

if typing.TYPE_CHECKING:
    from .evaluator_log_response import EvaluatorLogResponseParams
    from .log_response import LogResponseParams


class FlowLogResponseParams(typing_extensions.TypedDict):
    """
    General request for creating a Log
    """

    start_time: typing_extensions.NotRequired[dt.datetime]
    """
    When the logged event started.
    """

    end_time: typing_extensions.NotRequired[dt.datetime]
    """
    When the logged event ended.
    """

    output: typing_extensions.NotRequired[str]
    """
    Generated output from your model for the provided inputs. Can be `None` if logging an error, or if creating a parent Log with the intention to populate it later.
    """

    created_at: typing_extensions.NotRequired[dt.datetime]
    """
    User defined timestamp for when the log was created.
    """

    error: typing_extensions.NotRequired[str]
    """
    Error message if the log is an error.
    """

    provider_latency: typing_extensions.NotRequired[float]
    """
    Duration of the logged event in seconds.
    """

    stdout: typing_extensions.NotRequired[str]
    """
    Captured log and debug statements.
    """

    provider_request: typing_extensions.NotRequired[typing.Dict[str, typing.Optional[typing.Any]]]
    """
    Raw request sent to provider.
    """

    provider_response: typing_extensions.NotRequired[typing.Dict[str, typing.Optional[typing.Any]]]
    """
    Raw response received the provider.
    """

    inputs: typing_extensions.NotRequired[typing.Dict[str, typing.Optional[typing.Any]]]
    """
    The inputs passed to the Flow Log.
    """

    source: typing_extensions.NotRequired[str]
    """
    Identifies where the model was called from.
    """

    metadata: typing_extensions.NotRequired[typing.Dict[str, typing.Optional[typing.Any]]]
    """
    Any additional metadata to record.
    """

    source_datapoint_id: typing_extensions.NotRequired[str]
    """
    Unique identifier for the Datapoint that this Log is derived from. This can be used by Humanloop to associate Logs to Evaluations. If provided, Humanloop will automatically associate this Log to Evaluations that require a Log for this Datapoint-Version pair.
    """

    trace_parent_id: typing_extensions.NotRequired[str]
    """
    The ID of the parent Log to nest this Log under in a Trace.
    """

    batches: typing_extensions.NotRequired[typing.Sequence[str]]
    """
    Array of Batch Ids that this log is part of. Batches are used to group Logs together for offline Evaluations
    """

    user: typing_extensions.NotRequired[str]
    """
    End-user ID related to the Log.
    """

    environment: typing_extensions.NotRequired[str]
    """
    The name of the Environment the Log is associated to.
    """

    save: typing_extensions.NotRequired[bool]
    """
    Whether the request/response payloads will be stored on Humanloop.
    """

    id: str
    """
    Unique identifier for the Log.
    """

    evaluator_logs: typing.Sequence["EvaluatorLogResponseParams"]
    """
    List of Evaluator Logs associated with the Log. These contain Evaluator judgments on the Log.
    """

    trace_flow_id: typing_extensions.NotRequired[str]
    """
    Identifier for the Flow that the Trace belongs to.
    """

    trace_id: typing_extensions.NotRequired[str]
    """
    Identifier for the Trace that the Log belongs to.
    """

    trace_children: typing_extensions.NotRequired[typing.Sequence["LogResponseParams"]]
    """
    Logs nested under this Log in the Trace.
    """

    flow: FlowResponseParams
    """
    Flow used to generate the Log.
    """

    trace_status: typing_extensions.NotRequired[TraceStatus]
    """
    Status of the Trace. When a Trace is marked as `complete`, no more Logs can be added to it. Monitoring Evaluators will only run on completed Traces.
    """