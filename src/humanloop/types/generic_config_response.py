# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic

from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel
from .base_models_user_response import BaseModelsUserResponse


class GenericConfigResponse(UncheckedBaseModel):
    id: str = pydantic.Field()
    """
    String ID of config. Starts with `config_`.
    """

    other: typing.Optional[typing.Dict[str, typing.Any]] = pydantic.Field(default=None)
    """
    Other parameters that define the config.
    """

    type: typing.Literal["generic"] = "generic"
    created_by: typing.Optional[BaseModelsUserResponse] = pydantic.Field(default=None)
    """
    The user who created the config.
    """

    status: str = pydantic.Field()
    """
    Whether the config is committed or not.
    """

    name: str = pydantic.Field()
    """
    Name of config.
    """

    description: typing.Optional[str] = pydantic.Field(default=None)
    """
    Description of config.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
