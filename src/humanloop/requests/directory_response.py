# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import typing_extensions
import datetime as dt


class DirectoryResponseParams(typing_extensions.TypedDict):
    id: str
    """
    String ID of directory. Starts with `dir_`.
    """

    parent_id: typing_extensions.NotRequired[str]
    """
    ID of the parent directory. Will be `None` if the directory is the root directory. Starts with `dir_`.
    """

    name: str
    """
    Name of the directory.
    """

    path: str
    """
    Path to the directory, relative to the root directory. Includes name, e.g. `path/to/directory`.
    """

    created_at: dt.datetime
    updated_at: dt.datetime