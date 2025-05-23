# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel
from .log_status import LogStatus


class CreateFlowLogResponse(UncheckedBaseModel):
    """
    Response for a Flow Log.
    """

    id: str = pydantic.Field()
    """
    Unique identifier for the Log.
    """

    flow_id: str = pydantic.Field()
    """
    Unique identifier for the Flow.
    """

    version_id: str = pydantic.Field()
    """
    Unique identifier for the Flow Version.
    """

    log_status: typing.Optional[LogStatus] = pydantic.Field(default=None)
    """
    Status of the Flow Log. When a Flow Log is marked as `complete`, no more Logs can be added to it.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
