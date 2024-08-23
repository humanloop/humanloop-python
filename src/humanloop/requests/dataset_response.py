# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import typing_extensions
import typing
from .environment_response import EnvironmentResponseParams
import datetime as dt
from ..types.user_response import UserResponse
from ..types.version_status import VersionStatus
from .datapoint_response import DatapointResponseParams


class DatasetResponseParams(typing_extensions.TypedDict):
    """
    Base type that all File Responses should inherit from.

    Attributes defined here are common to all File Responses and should be overridden
    in the inheriting classes with documentation and appropriate Field definitions.
    """

    path: str
    """
    Path of the Dataset, including the name, which is used as a unique identifier.
    """

    id: str
    """
    Unique identifier for the Dataset. Starts with `ds_`.
    """

    directory_id: typing_extensions.NotRequired[str]
    """
    ID of the directory that the file is in on Humanloop.
    """

    name: str
    """
    Name of the Dataset, which is used as a unique identifier.
    """

    version_id: str
    """
    Unique identifier for the specific Dataset Version. If no query params provided, the default deployed Dataset Version is returned. Starts with `dsv_`.
    """

    type: typing_extensions.NotRequired[typing.Literal["dataset"]]
    environments: typing_extensions.NotRequired[typing.Sequence[EnvironmentResponseParams]]
    """
    The list of environments the Dataset Version is deployed to.
    """

    created_at: dt.datetime
    updated_at: dt.datetime
    created_by: typing_extensions.NotRequired[UserResponse]
    """
    The user who created the Dataset.
    """

    status: VersionStatus
    """
    The status of the Dataset Version.
    """

    last_used_at: dt.datetime
    commit_message: typing_extensions.NotRequired[str]
    """
    Message describing the changes made. If provided, a committed version of the Dataset is created. Otherwise, an uncommitted version is created.
    """

    datapoints_count: int
    """
    The number of Datapoints in this Dataset version.
    """

    datapoints: typing_extensions.NotRequired[typing.Sequence[DatapointResponseParams]]
    """
    The list of Datapoints in this Dataset version. Only provided if explicitly requested.
    """
