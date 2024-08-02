# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions

from ..types.evaluator_arguments_type import EvaluatorArgumentsType
from ..types.evaluator_return_type_enum import EvaluatorReturnTypeEnum
from .base_models_user_response import BaseModelsUserResponseParams


class EvaluatorConfigResponseParams(typing_extensions.TypedDict):
    id: str
    """
    String ID of config. Starts with `config_`.
    """

    other: typing_extensions.NotRequired[typing.Dict[str, typing.Any]]
    """
    Other parameters that define the config.
    """

    type: typing.Literal["evaluator"]
    created_by: typing_extensions.NotRequired[BaseModelsUserResponseParams]
    """
    The user who created the config.
    """

    status: str
    """
    Whether the config is committed or not.
    """

    name: str
    """
    Name of config.
    """

    description: typing_extensions.NotRequired[str]
    """
    Description of config.
    """

    evaluator_type: str
    """
    Type of evaluator.
    """

    code: typing_extensions.NotRequired[str]
    """
    The code for the evaluator. This code will be executed in a sandboxed environment.
    """

    arguments_type: typing_extensions.NotRequired[EvaluatorArgumentsType]
    """
    Whether this evaluator is target-free or target-required.
    """

    return_type: typing_extensions.NotRequired[EvaluatorReturnTypeEnum]
    """
    The type of the return value of the evaluator.
    """