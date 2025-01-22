# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import typing_extensions
import datetime as dt
import typing
from .chat_message import ChatMessageParams
from .prompt_call_response_tool_choice import PromptCallResponseToolChoiceParams
from .prompt_response import PromptResponseParams
from .prompt_call_log_response import PromptCallLogResponseParams


class PromptCallResponseParams(typing_extensions.TypedDict):
    """
    Response model for a Prompt call with potentially multiple log samples.
    """

    start_time: typing_extensions.NotRequired[dt.datetime]
    """
    When the logged event started.
    """

    end_time: typing_extensions.NotRequired[dt.datetime]
    """
    When the logged event ended.
    """

    messages: typing_extensions.NotRequired[typing.Sequence[ChatMessageParams]]
    """
    The messages passed to the to provider chat endpoint.
    """

    tool_choice: typing_extensions.NotRequired[PromptCallResponseToolChoiceParams]
    """
    Controls how the model uses tools. The following options are supported: 
    - `'none'` means the model will not call any tool and instead generates a message; this is the default when no tools are provided as part of the Prompt. 
    - `'auto'` means the model can decide to call one or more of the provided tools; this is the default when tools are provided as part of the Prompt. 
    - `'required'` means the model can decide to call one or more of the provided tools. 
    - `{'type': 'function', 'function': {name': <TOOL_NAME>}}` forces the model to use the named function.
    """

    prompt: PromptResponseParams
    """
    Prompt used to generate the Log.
    """

    inputs: typing_extensions.NotRequired[typing.Dict[str, typing.Optional[typing.Any]]]
    """
    The inputs passed to the prompt template.
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

    log_id: typing_extensions.NotRequired[str]
    """
    This will identify a Log. If you don't provide a Log ID, Humanloop will generate one for you.
    """

    id: str
    """
    ID of the log.
    """

    trace_id: typing_extensions.NotRequired[str]
    """
    ID of the Trace containing the Prompt Call Log.
    """

    logs: typing.Sequence[PromptCallLogResponseParams]
    """
    The logs generated by the Prompt call.
    """
