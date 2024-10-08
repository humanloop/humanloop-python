# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import datetime as dt


class EvaluatorAggregateParams(typing_extensions.TypedDict):
    value: float
    """
    The aggregated value of the evaluator.
    """

    evaluator_id: str
    """
    ID of the evaluator.
    """

    evaluator_version_id: str
    """
    ID of the evaluator version.
    """

    created_at: dt.datetime
    updated_at: dt.datetime
