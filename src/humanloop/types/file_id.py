# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
import typing
import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class FileId(UncheckedBaseModel):
    """
    Specification of a File by its ID.
    """

    environment: typing.Optional[str] = pydantic.Field(default=None)
    """
    If provided, the Version deployed to this Environment is used. If not provided, the Version deployed to the default Environment is used.
    """

    id: str = pydantic.Field()
    """
    Unique identifier for the File.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow