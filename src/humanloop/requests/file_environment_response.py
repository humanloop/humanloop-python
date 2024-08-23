# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import datetime as dt
from ..types.environment_tag import EnvironmentTag
import typing_extensions
from .file_environment_response_file import FileEnvironmentResponseFileParams


class FileEnvironmentResponseParams(typing_extensions.TypedDict):
    """
    Response model for the List Environments endpoint under Files.

    Contains the deployed version of the File, if one is deployed to the Environment.
    """

    id: str
    created_at: dt.datetime
    name: str
    tag: EnvironmentTag
    file: typing_extensions.NotRequired[FileEnvironmentResponseFileParams]
    """
    The version of the File that is deployed to the Environment, if one is deployed.
    """
