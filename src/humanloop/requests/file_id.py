# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import typing_extensions


class FileIdParams(typing_extensions.TypedDict):
    """
    Specification of a File by its ID.
    """

    environment: typing_extensions.NotRequired[str]
    """
    If provided, the Version deployed to this Environment is used. If not provided, the Version deployed to the default Environment is used.
    """

    id: str
    """
    Unique identifier for the File.
    """
