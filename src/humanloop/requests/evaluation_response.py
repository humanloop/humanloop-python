# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

import typing_extensions

from ..types.evaluation_status import EvaluationStatus
from .dataset_response import DatasetResponseParams
from .evaluatee_response import EvaluateeResponseParams
from .evaluation_evaluator_response import EvaluationEvaluatorResponseParams
from .user_response import UserResponseParams


class EvaluationResponseParams(typing_extensions.TypedDict):
    id: str
    """
    Unique identifier for the Evaluation. Starts with `evr`.
    """

    dataset: DatasetResponseParams
    """
    The Dataset used in the Evaluation.
    """

    evaluatees: typing.Sequence[EvaluateeResponseParams]
    """
    The Prompt/Tool Versions included in the Evaluation.
    """

    evaluators: typing.Sequence[EvaluationEvaluatorResponseParams]
    """
    The Evaluator Versions used to evaluate.
    """

    status: EvaluationStatus
    """
    The current status of the Evaluation.
    
    - `"pending"`: The Evaluation has been created but is not actively being worked on by Humanloop.
    - `"running"`: Humanloop is checking for any missing Logs and Evaluator Logs, and will generate them where appropriate.
    - `"completed"`: All Logs an Evaluator Logs have been generated.
    - `"cancelled"`: The Evaluation has been cancelled by the user. Humanloop will stop generating Logs and Evaluator Logs.
    """

    created_at: dt.datetime
    created_by: typing_extensions.NotRequired[UserResponseParams]
    updated_at: dt.datetime