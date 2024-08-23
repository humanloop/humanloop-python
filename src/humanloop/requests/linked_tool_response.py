# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions


class LinkedToolResponseParams(typing_extensions.TypedDict):
    name: str
    """
    Name for the tool referenced by the model.
    """

    description: str
    """
    Description of the tool referenced by the model
    """

    parameters: typing_extensions.NotRequired[typing.Dict[str, typing.Any]]
    """
    Parameters needed to run the Tool, defined in JSON Schema format: https://json-schema.org/
    """

    id: str
    """
    Unique identifier for the Tool linked.
    """

    version_id: str
    """
    Unique identifier for the Tool Version linked.
    """
