# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations

import typing

import typing_extensions

if typing.TYPE_CHECKING:
    from .version_id_response_version import VersionIdResponseVersionParams


class VersionIdResponseParams(typing_extensions.TypedDict):
    """
    A reference to a specific Version by its ID
    """

    version: "VersionIdResponseVersionParams"
    """
    The specific Version being referenced.
    """

    type: typing.Literal["version"]