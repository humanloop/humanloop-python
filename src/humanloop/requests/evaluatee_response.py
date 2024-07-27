# This file was auto-generated by Fern from our API Definition.

import typing_extensions

from .evaluated_version_response import EvaluatedVersionResponseParams


class EvaluateeResponseParams(typing_extensions.TypedDict):
    """
    Version of the Evaluatee being evaluated.
    """

    version: EvaluatedVersionResponseParams
    batch_id: typing_extensions.NotRequired[str]
    """
    Unique identifier for the batch of Logs to include in the Evaluation Report.
    """

    orchestrated: bool
    """
    Whether the Prompt/Tool is orchestrated by Humanloop. Default is `True`. If `False`, a log for the Prompt/Tool should be submitted by the user via the API.
    """
