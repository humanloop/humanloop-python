# This file was auto-generated by Fern from our API Definition.

import typing
from .tool_choice import ToolChoice

PromptLogResponseToolChoice = typing.Union[
    typing.Literal["none"], typing.Literal["auto"], typing.Literal["required"], ToolChoice
]
