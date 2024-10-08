# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import typing_extensions
from .chat_message_content import ChatMessageContentParams
from ..types.chat_role import ChatRole
import typing
from .tool_call import ToolCallParams


class ChatMessageParams(typing_extensions.TypedDict):
    content: typing_extensions.NotRequired[ChatMessageContentParams]
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
