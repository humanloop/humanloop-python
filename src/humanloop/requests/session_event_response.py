# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations
import typing_extensions
from .log_response import LogResponseParams
import typing


class SessionEventResponseParams(typing_extensions.TypedDict):
    log: LogResponseParams
    children: typing.Sequence["SessionEventResponseParams"]