# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel
from .tool_function import ToolFunction


class ToolKernelRequest(UncheckedBaseModel):
    function: typing.Optional[ToolFunction] = pydantic.Field(default=None)
    """
    Callable function specification of the Tool shown to the model for tool calling.
    """

    source_code: typing.Optional[str] = pydantic.Field(default=None)
    """
    Code source of the Tool.
    """

    setup_values: typing.Optional[typing.Dict[str, typing.Optional[typing.Any]]] = pydantic.Field(default=None)
    """
    Values needed to setup the Tool, defined in JSON Schema format: https://json-schema.org/
    """

    attributes: typing.Optional[typing.Dict[str, typing.Optional[typing.Any]]] = pydantic.Field(default=None)
    """
    Additional fields to describe the Tool. Helpful to separate Tool versions from each other with details on how they were created or used.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
