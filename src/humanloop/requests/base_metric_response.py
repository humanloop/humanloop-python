# This file was auto-generated by Fern from our API Definition.

import datetime as dt

import typing_extensions


class BaseMetricResponseParams(typing_extensions.TypedDict):
    id: str
    """
    ID of the metric. Starts with 'metric\_'.
    """

    name: str
    """
    The name of the metric.
    """

    description: str
    """
    A description of what the metric measures.
    """

    code: str
    """
    Python code used to calculate a metric value on each logged datapoint.
    """

    default: bool
    """
    Whether the metric is a global default metric. Metrics with this flag enabled cannot be deleted or modified.
    """

    active: bool
    """
    If enabled, the metric is calculated for every logged datapoint.
    """

    created_at: dt.datetime
    updated_at: dt.datetime
