# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ..core.datetime_utils import serialize_datetime
from ..core.pydantic_utilities import deep_union_pydantic_dicts, pydantic_v1
from ..core.unchecked_base_model import UncheckedBaseModel
from .datapoint_response import DatapointResponse
from .environment_response import EnvironmentResponse
from .user_response import UserResponse
from .version_status import VersionStatus


class DatasetResponse(UncheckedBaseModel):
    """
    Base type that all File Responses should inherit from.

    Attributes defined here are common to all File Responses and should be overridden
    in the inheriting classes with documentation and appropriate Field definitions.
    """

    id: str = pydantic_v1.Field()
    """
    Unique identifier for the Dataset. Starts with `ds_`.
    """

    name: str = pydantic_v1.Field()
    """
    Name of the Dataset, which is used as a unique identifier.
    """

    version_id: str = pydantic_v1.Field()
    """
    Unique identifier for the specific Dataset Version. If no query params provided, the default deployed Dataset Version is returned. Starts with `dsv_`.
    """

    directory_id: typing.Optional[str] = None
    environments: typing.Optional[typing.List[EnvironmentResponse]] = pydantic_v1.Field(default=None)
    """
    The list of environments the Dataset Version is deployed to.
    """

    created_at: dt.datetime
    updated_at: dt.datetime
    created_by: typing.Optional[UserResponse] = pydantic_v1.Field(default=None)
    """
    The user who created the Dataset.
    """

    status: VersionStatus = pydantic_v1.Field()
    """
    The status of the Dataset Version.
    """

    last_used_at: dt.datetime
    path: str = pydantic_v1.Field()
    """
    Path of the Dataset, including the name, which is used as a unique identifier.
    """

    commit_message: typing.Optional[str] = pydantic_v1.Field(default=None)
    """
    Message describing the changes made. If provided, a committed version of the Dataset is created. Otherwise, an uncommitted version is created.
    """

    datapoints_count: int = pydantic_v1.Field()
    """
    The number of Datapoints in this Dataset version.
    """

    datapoints: typing.Optional[typing.List[DatapointResponse]] = pydantic_v1.Field(default=None)
    """
    The list of Datapoints in this Dataset version. Only provided if explicitly requested.
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
