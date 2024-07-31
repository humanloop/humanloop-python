# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions

from .chat_message import ChatMessageParams
from .datapoint_response_target_value import DatapointResponseTargetValueParams


class DatapointResponseParams(typing_extensions.TypedDict):
    inputs: typing_extensions.NotRequired[typing.Dict[str, str]]
    """
    The inputs to the prompt template.
    """

    messages: typing_extensions.NotRequired[typing.Sequence[ChatMessageParams]]
    """
    List of chat messages to provide to the model.
    """

    target: typing_extensions.NotRequired[typing.Dict[str, DatapointResponseTargetValueParams]]
    """
    Object with criteria necessary to evaluate generations with this Datapoint. This is passed in as an argument to Evaluators when used in an Evaluation.
    """

    id: str
    """
    Unique identifier for the Datapoint. Starts with `dp_`.
    """
