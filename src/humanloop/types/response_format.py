# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
from .response_format_type import ResponseFormatType
import typing
import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class ResponseFormat(UncheckedBaseModel):
    """
    Response format of the model.
    """

    type: ResponseFormatType
    json_schema: typing.Optional[typing.Dict[str, typing.Optional[typing.Any]]] = pydantic.Field(default=None)
    """
    The JSON schema of the response format if type is json_schema.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
