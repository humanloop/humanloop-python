# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import typing
from .evaluation_log_response import EvaluationLogResponseParams


class PaginatedDataEvaluationLogResponseParams(typing_extensions.TypedDict):
    records: typing.Sequence[EvaluationLogResponseParams]
    page: int
    size: int
    total: int