# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations
import typing_extensions
import typing_extensions
import typing
from .environment_response import EnvironmentResponseParams
import datetime as dt
from ..types.user_response import UserResponse
from ..types.version_status import VersionStatus
from .evaluator_aggregate import EvaluatorAggregateParams
import typing

if typing.TYPE_CHECKING:
    from .monitoring_evaluator_response import MonitoringEvaluatorResponseParams


class FlowResponseParams(typing_extensions.TypedDict):
    """
    Response model for a Flow.
    """

    path: str
    """
    Path of the Flow, including the name, which is used as a unique identifier.
    """

    id: str
    """
    Unique identifier for the Flow. Starts with fl\_.
    """

    directory_id: typing_extensions.NotRequired[str]
    """
    ID of the directory that the file is in on Humanloop.
    """

    attributes: typing.Dict[str, typing.Optional[typing.Any]]
    """
    A key-value object identifying the Flow Version.
    """

    commit_message: typing_extensions.NotRequired[str]
    """
    Message describing the changes made. If provided, a committed version of the Flow is created. Otherwise, an uncommitted version is created.
    """

    name: str
    """
    Name of the Flow.
    """

    version_id: str
    """
    Unique identifier for the specific Flow Version. If no query params provided, the default deployed Flow Version is returned.
    """

    type: typing_extensions.NotRequired[typing.Literal["flow"]]
    environments: typing_extensions.NotRequired[typing.Sequence[EnvironmentResponseParams]]
    """
    The list of environments the Flow Version is deployed to.
    """

    created_at: dt.datetime
    updated_at: dt.datetime
    created_by: typing_extensions.NotRequired[UserResponse]
    """
    The user who created the Flow.
    """

    status: VersionStatus
    """
    The status of the Flow Version.
    """

    last_used_at: dt.datetime
    version_logs_count: int
    """
    The number of logs that have been generated for this Flow Version
    """

    evaluator_aggregates: typing_extensions.NotRequired[typing.Sequence[EvaluatorAggregateParams]]
    """
    Aggregation of Evaluator results for the Flow Version.
    """

    evaluators: typing_extensions.NotRequired[typing.Sequence["MonitoringEvaluatorResponseParams"]]
    """
    The list of Monitoring Evaluators associated with the Flow Version.
    """
