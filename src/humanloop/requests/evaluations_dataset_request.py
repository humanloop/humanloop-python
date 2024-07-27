# This file was auto-generated by Fern from our API Definition.

import typing_extensions


class EvaluationsDatasetRequestParams(typing_extensions.TypedDict):
    version_id: str
    """
    Unique identifier for the Dataset Version to use in this evaluation. Starts with `dsv_`.
    """
