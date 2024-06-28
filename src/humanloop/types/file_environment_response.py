# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ..core.datetime_utils import serialize_datetime
from ..core.pydantic_utilities import deep_union_pydantic_dicts, pydantic_v1
from ..core.unchecked_base_model import UncheckedBaseModel
from .environment_tag import EnvironmentTag
from .file_environment_response_file import FileEnvironmentResponseFile


class FileEnvironmentResponse(UncheckedBaseModel):
    """
    Response model for the List Environments endpoint under Files.

    Contains the deployed version of the File, if one is deployed to the Environment.
    """

    id: str
    created_at: dt.datetime
    name: str
    tag: EnvironmentTag
    file: typing.Optional[FileEnvironmentResponseFile] = pydantic_v1.Field(default=None)
    """
    The version of the File that is deployed to the Environment, if one is deployed.
    """

    def json(self, **kwargs: typing.Any) -> str:
        kwargs_with_defaults: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        return super().json(**kwargs_with_defaults)

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        kwargs_with_defaults_exclude_unset: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        kwargs_with_defaults_exclude_none: typing.Any = {"by_alias": True, "exclude_none": True, **kwargs}

        return deep_union_pydantic_dicts(
            super().dict(**kwargs_with_defaults_exclude_unset), super().dict(**kwargs_with_defaults_exclude_none)
        )

    class Config:
        frozen = True
        smart_union = True
        extra = pydantic_v1.Extra.allow
        json_encoders = {dt.datetime: serialize_datetime}