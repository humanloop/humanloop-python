# This file was auto-generated by Fern from our API Definition.

import typing_extensions


class MonitoringEvaluatorEnvironmentRequestParams(typing_extensions.TypedDict):
    evaluator_id: str
    """
    Unique identifier for the Evaluator to be used for monitoring.
    """

    environment_id: str
    """
    Unique identifier for the Environment. The Evaluator Version deployed to this Environment will be used for monitoring.
    """
