# This file was auto-generated by Fern from our API Definition.

import typing_extensions

from ..types.chat_tool_type import ChatToolType
from .function_tool_choice import FunctionToolChoiceParams


class ToolChoiceParams(typing_extensions.TypedDict):
    """
    Tool choice to force the model to use a tool.
    """

    type: ChatToolType
    function: FunctionToolChoiceParams
