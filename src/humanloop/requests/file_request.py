# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import typing_extensions


class FileRequestParams(typing_extensions.TypedDict):
    id: typing_extensions.NotRequired[str]
    """
    ID for an existing File.
    """

    path: typing_extensions.NotRequired[str]
    """
    Path of the File, including the name. This locates the File in the Humanloop filesystem and is used as as a unique identifier. For example: `folder/name` or just `name`.
    """
