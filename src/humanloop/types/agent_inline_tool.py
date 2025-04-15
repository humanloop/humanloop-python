# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
import typing
from .tool_function import ToolFunction
from .stop_agent_enum import StopAgentEnum
from ..core.pydantic_utilities import IS_PYDANTIC_V2
import pydantic


class AgentInlineTool(UncheckedBaseModel):
    type: typing.Literal["inline"] = "inline"
    json_schema: ToolFunction
    stop_agent: StopAgentEnum

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
