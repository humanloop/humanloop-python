# This file was auto-generated by Fern from our API Definition.

import typing_extensions


class CreatePromptLogResponseParams(typing_extensions.TypedDict):
    id: str
    """
    String ID of log.
    """

    prompt_id: str
    """
    ID of the Prompt the log belongs to.
    """

    version_id: str
    """
    ID of the specific version of the Prompt.
    """

    session_id: typing_extensions.NotRequired[str]
    """
    String ID of session the log belongs to.
    """