# This file was auto-generated by Fern from our API Definition.

import typing
from ...requests.tool_choice import ToolChoiceParams

PromptLogUpdateRequestToolChoiceParams = typing.Union[
    typing.Literal["none"], typing.Literal["auto"], typing.Literal["required"], ToolChoiceParams
]
