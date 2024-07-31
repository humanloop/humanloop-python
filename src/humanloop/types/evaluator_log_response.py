# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

import pydantic

from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel
from .evaluator_response import EvaluatorResponse


class EvaluatorLogResponse(UncheckedBaseModel):
    """
    General request for creating a Log
    """

    output: typing.Optional[str] = pydantic.Field(default=None)
    """
    Generated output from your model for the provided inputs. Can be `None` if logging an error, or if creating a parent Log with the intention to populate it later.
    """

    created_at: typing.Optional[dt.datetime] = pydantic.Field(default=None)
    """
    User defined timestamp for when the log was created.
    """

    error: typing.Optional[str] = pydantic.Field(default=None)
    """
    Error message if the log is an error.
    """

    provider_latency: typing.Optional[float] = pydantic.Field(default=None)
    """
    Duration of the logged event in seconds.
    """

    provider_request: typing.Optional[typing.Dict[str, typing.Any]] = pydantic.Field(default=None)
    """
    Raw request sent to provider.
    """

    provider_response: typing.Optional[typing.Dict[str, typing.Any]] = pydantic.Field(default=None)
    """
    Raw response received the provider.
    """

    session_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    Unique identifier for the Session to associate the Log to. Allows you to record multiple Logs to a Session (using an ID kept by your internal systems) by passing the same `session_id` in subsequent log requests.
    """

    parent_id: str = pydantic.Field()
    """
    Identifier of the evaluated Log. The newly created Log will have this one set as parent.
    """

    inputs: typing.Optional[typing.Dict[str, typing.Any]] = pydantic.Field(default=None)
    """
    The inputs passed to the prompt template.
    """

    source: typing.Optional[str] = pydantic.Field(default=None)
    """
    Identifies where the model was called from.
    """

    metadata: typing.Optional[typing.Dict[str, typing.Any]] = pydantic.Field(default=None)
    """
    Any additional metadata to record.
    """

    save: typing.Optional[bool] = pydantic.Field(default=None)
    """
    Whether the request/response payloads will be stored on Humanloop.
    """

    source_datapoint_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    Unique identifier for the Datapoint that this Log is derived from. This can be used by Humanloop to associate Logs to Evaluations. If provided, Humanloop will automatically associate this Log to Evaluations that require a Log for this Datapoint-Version pair.
    """

    batches: typing.Optional[typing.List[str]] = pydantic.Field(default=None)
    """
    Array of Batch Ids that this log is part of. Batches are used to group Logs together for offline Evaluations
    """

    user: typing.Optional[str] = pydantic.Field(default=None)
    """
    End-user ID related to the Log.
    """

    environment: typing.Optional[str] = pydantic.Field(default=None)
    """
    The name of the Environment the Log is associated to.
    """

    judgment: typing.Optional[typing.Any] = None
    id: str = pydantic.Field()
    """
    Unique identifier for the Log.
    """

    evaluator: EvaluatorResponse = pydantic.Field()
    """
    The Evaluator used to generate the judgment.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
