# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions


class ConfigToolResponseParams(typing_extensions.TypedDict):
    id: str
    """
    The ID of the tool. Starts with either `config_` or `oc_`.
    """

    name: str
    """
    Name for the tool referenced by the model.
    """

    description: typing_extensions.NotRequired[str]
    """
    Description of the tool referenced by the model
    """

    parameters: typing_extensions.NotRequired[typing.Dict[str, typing.Any]]
    """
    Definition of parameters needed to run the tool. Provided in jsonschema format: https://json-schema.org/
    """

    source: typing_extensions.NotRequired[str]
    """
    The origin of the tool
    """
