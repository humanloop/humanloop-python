# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import typing
from .datapoint_response import DatapointResponseParams


class PaginatedDatapointResponseParams(typing_extensions.TypedDict):
    records: typing.Sequence[DatapointResponseParams]
    page: int
    size: int
    total: int
