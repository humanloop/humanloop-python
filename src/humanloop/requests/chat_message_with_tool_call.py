# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions

from ..types.chat_role import ChatRole
from .chat_message_with_tool_call_content import ChatMessageWithToolCallContentParams
from .function_tool import FunctionToolParams
from .tool_call import ToolCallParams


class ChatMessageWithToolCallParams(typing_extensions.TypedDict):
    content: typing_extensions.NotRequired[ChatMessageWithToolCallContentParams]
    """
    The content of the message.
    """

    name: typing_extensions.NotRequired[str]
    """
    Optional name of the message author.
    """

    tool_call_id: typing_extensions.NotRequired[str]
    """
    Tool call that this message is responding to.
    """

    role: ChatRole
    """
    Role of the message author.
    """

    tool_calls: typing_extensions.NotRequired[typing.Sequence[ToolCallParams]]
    """
    A list of tool calls requested by the assistant.
    """

    tool_call: typing_extensions.NotRequired[FunctionToolParams]
    """
    NB: Deprecated in favour of tool_calls. A tool call requested by the assistant.
    """
