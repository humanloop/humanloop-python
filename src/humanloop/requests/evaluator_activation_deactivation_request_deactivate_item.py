# This file was auto-generated by Fern from our API Definition.

import typing

from .monitoring_evaluator_environment_request import MonitoringEvaluatorEnvironmentRequestParams
from .monitoring_evaluator_version_request import MonitoringEvaluatorVersionRequestParams

EvaluatorActivationDeactivationRequestDeactivateItemParams = typing.Union[
    MonitoringEvaluatorVersionRequestParams, MonitoringEvaluatorEnvironmentRequestParams
]
