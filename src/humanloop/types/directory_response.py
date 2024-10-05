# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
import pydantic
import typing
import datetime as dt
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class DirectoryResponse(UncheckedBaseModel):
    id: str = pydantic.Field()
    """
    String ID of directory. Starts with `dir_`.
    """

    parent_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    ID of the parent directory. Will be `None` if the directory is the root directory. Starts with `dir_`.
    """

    name: str = pydantic.Field()
    """
    Name of the directory.
    """

    path: str = pydantic.Field()
    """
    Path to the directory, relative to the root directory. Includes name, e.g. `path/to/directory`.
    """

    created_at: dt.datetime
    updated_at: dt.datetime

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow