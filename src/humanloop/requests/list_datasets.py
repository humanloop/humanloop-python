# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import typing
from .dataset_response import DatasetResponseParams


class ListDatasetsParams(typing_extensions.TypedDict):
    records: typing.Sequence[DatasetResponseParams]
    """
    The list of Datasets.
    """
