# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import typing
from .prompt_response import PromptResponseParams


class PaginatedDataPromptResponseParams(typing_extensions.TypedDict):
    records: typing.Sequence[PromptResponseParams]
    page: int
    size: int
    total: int
