# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations

import datetime as dt
import typing

import typing_extensions
from ..types.monitoring_evaluator_state import MonitoringEvaluatorState

if typing.TYPE_CHECKING:
    from .evaluator_response import EvaluatorResponseParams
    from .version_reference_response import VersionReferenceResponseParams


class MonitoringEvaluatorResponseParams(typing_extensions.TypedDict):
    version_reference: "VersionReferenceResponseParams"
    """
    The Evaluator Version used for monitoring. This can be a specific Version by ID, or a Version deployed to an Environment.
    """

    version: typing_extensions.NotRequired["EvaluatorResponseParams"]
    """
    The deployed Version.
    """

    state: MonitoringEvaluatorState
    """
    The state of the Monitoring Evaluator. Either `active` or `inactive`
    """

    created_at: dt.datetime
    updated_at: dt.datetime
