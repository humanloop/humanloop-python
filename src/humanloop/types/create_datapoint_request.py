# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic

from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel
from .chat_message import ChatMessage
from .create_datapoint_request_target_value import CreateDatapointRequestTargetValue


class CreateDatapointRequest(UncheckedBaseModel):
    inputs: typing.Optional[typing.Dict[str, str]] = pydantic.Field(default=None)
    """
    The inputs to the prompt template.
    """

    messages: typing.Optional[typing.List[ChatMessage]] = pydantic.Field(default=None)
    """
    List of chat messages to provide to the model.
    """

    target: typing.Optional[typing.Dict[str, CreateDatapointRequestTargetValue]] = pydantic.Field(default=None)
    """
    Object with criteria necessary to evaluate generations with this Datapoint. This is passed in as an argument to Evaluators when used in an Evaluation.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
