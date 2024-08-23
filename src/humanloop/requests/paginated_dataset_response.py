# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions

from .dataset_response import DatasetResponseParams


class PaginatedDatasetResponseParams(typing_extensions.TypedDict):
    records: typing.Sequence[DatasetResponseParams]
    page: int
    size: int
    total: int
