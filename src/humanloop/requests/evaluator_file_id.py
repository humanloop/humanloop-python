# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import typing_extensions


class EvaluatorFileIdParams(typing_extensions.TypedDict):
    """
    Base model for specifying an Evaluator for an Evaluation.
    """

    environment: typing_extensions.NotRequired[str]
    """
    If provided, the Version deployed to this Environment is used. If not provided, the Version deployed to the default Environment is used.
    """

    id: str
    """
    Unique identifier for the File.
    """

    orchestrated: typing_extensions.NotRequired[bool]
    """
    Whether the Evaluator is orchestrated by Humanloop. Default is `True`. If `False`, a log for the Evaluator should be submitted by the user via the API.
    """