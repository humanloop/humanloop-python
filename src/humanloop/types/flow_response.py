# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations
from ..core.unchecked_base_model import UncheckedBaseModel
import pydantic
import typing
from .environment_response import EnvironmentResponse
import datetime as dt
from .user_response import UserResponse
from .version_status import VersionStatus
from .evaluator_aggregate import EvaluatorAggregate
from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.pydantic_utilities import update_forward_refs


class FlowResponse(UncheckedBaseModel):
    """
    Response model for a Flow.
    """

    path: str = pydantic.Field()
    """
    Path of the Flow, including the name, which is used as a unique identifier.
    """

    id: str = pydantic.Field()
    """
    Unique identifier for the Flow. Starts with fl\_.
    """

    directory_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    ID of the directory that the file is in on Humanloop.
    """

    attributes: typing.Dict[str, typing.Optional[typing.Any]] = pydantic.Field()
    """
    A key-value object identifying the Flow Version.
    """

    commit_message: typing.Optional[str] = pydantic.Field(default=None)
    """
    Message describing the changes made. If provided, a committed version of the Flow is created. Otherwise, an uncommitted version is created.
    """

    name: str = pydantic.Field()
    """
    Name of the Flow.
    """

    version_id: str = pydantic.Field()
    """
    Unique identifier for the specific Flow Version. If no query params provided, the default deployed Flow Version is returned.
    """

    type: typing.Optional[typing.Literal["flow"]] = None
    environments: typing.Optional[typing.List[EnvironmentResponse]] = pydantic.Field(default=None)
    """
    The list of environments the Flow Version is deployed to.
    """

    created_at: dt.datetime
    updated_at: dt.datetime
    created_by: typing.Optional[UserResponse] = pydantic.Field(default=None)
    """
    The user who created the Flow.
    """

    status: VersionStatus = pydantic.Field()
    """
    The status of the Flow Version.
    """

    last_used_at: dt.datetime
    version_logs_count: int = pydantic.Field()
    """
    The number of logs that have been generated for this Flow Version
    """

    evaluator_aggregates: typing.Optional[typing.List[EvaluatorAggregate]] = pydantic.Field(default=None)
    """
    Aggregation of Evaluator results for the Flow Version.
    """

    evaluators: typing.Optional[typing.List["MonitoringEvaluatorResponse"]] = pydantic.Field(default=None)
    """
    The list of Monitoring Evaluators associated with the Flow Version.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow


from .monitoring_evaluator_response import MonitoringEvaluatorResponse  # noqa: E402

update_forward_refs(FlowResponse)