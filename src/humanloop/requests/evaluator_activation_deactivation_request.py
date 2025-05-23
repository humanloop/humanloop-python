# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions
from .evaluator_activation_deactivation_request_activate_item import (
    EvaluatorActivationDeactivationRequestActivateItemParams,
)
from .evaluator_activation_deactivation_request_deactivate_item import (
    EvaluatorActivationDeactivationRequestDeactivateItemParams,
)


class EvaluatorActivationDeactivationRequestParams(typing_extensions.TypedDict):
    activate: typing_extensions.NotRequired[typing.Sequence[EvaluatorActivationDeactivationRequestActivateItemParams]]
    """
    Evaluators to activate for Monitoring. These will be automatically run on new Logs.
    """

    deactivate: typing_extensions.NotRequired[
        typing.Sequence[EvaluatorActivationDeactivationRequestDeactivateItemParams]
    ]
    """
    Evaluators to deactivate. These will not be run on new Logs.
    """
