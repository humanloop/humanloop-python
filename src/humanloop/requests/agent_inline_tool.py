# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions
from ..types.on_agent_call_enum import OnAgentCallEnum
from .tool_function import ToolFunctionParams


class AgentInlineToolParams(typing_extensions.TypedDict):
    type: typing.Literal["inline"]
    json_schema: ToolFunctionParams
    on_agent_call: typing_extensions.NotRequired[OnAgentCallEnum]
