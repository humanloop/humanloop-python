# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions


class NumericEvaluatorStatsResponseParams(typing_extensions.TypedDict):
    """
    Base attributes for stats for an Evaluator Version-Evaluated Version pair
    in the Evaluation.
    """

    evaluator_version_id: str
    """
    Unique identifier for the Evaluator Version.
    """

    total_logs: int
    """
    The total number of Logs generated by this Evaluator Version on the Evaluated Version's Logs. This includes Nulls and Errors.
    """

    num_judgments: int
    """
    The total number of Evaluator judgments for this Evaluator Version. This excludes Nulls and Errors.
    """

    num_nulls: int
    """
    The total number of null judgments (i.e. abstentions) for this Evaluator Version.
    """

    num_errors: int
    """
    The total number of errored Evaluators for this Evaluator Version.
    """

    mean: typing_extensions.NotRequired[float]
    sum: typing_extensions.NotRequired[float]
    std: typing_extensions.NotRequired[float]
    percentiles: typing.Dict[str, float]
