# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
import typing
from .session_response import SessionResponse
from ..core.pydantic_utilities import IS_PYDANTIC_V2
import pydantic


class PaginatedSessionResponse(UncheckedBaseModel):
    records: typing.List[SessionResponse]
    page: int
    size: int
    total: int

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
