# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions
from .evaluation_response import EvaluationResponseParams


class PaginatedEvaluationResponseParams(typing_extensions.TypedDict):
    records: typing.Sequence[EvaluationResponseParams]
    page: int
    size: int
    total: int
