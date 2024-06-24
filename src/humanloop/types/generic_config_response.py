# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ..core.datetime_utils import serialize_datetime
from ..core.pydantic_utilities import deep_union_pydantic_dicts, pydantic_v1
from ..core.unchecked_base_model import UncheckedBaseModel
from .base_models_user_response import BaseModelsUserResponse


class GenericConfigResponse(UncheckedBaseModel):
    id: str = pydantic_v1.Field()
    """
    String ID of config. Starts with `config_`.
    """

    other: typing.Optional[typing.Dict[str, typing.Any]] = pydantic_v1.Field(default=None)
    """
    Other parameters that define the config.
    """

    type: typing.Literal["generic"] = "generic"
    created_by: typing.Optional[BaseModelsUserResponse] = pydantic_v1.Field(default=None)
    """
    The user who created the config.
    """

    status: str = pydantic_v1.Field()
    """
    Whether the config is committed or not.
    """

    name: str = pydantic_v1.Field()
    """
    Name of config.
    """

    description: typing.Optional[str] = pydantic_v1.Field(default=None)
    """
    Description of config.
    """

    def json(self, **kwargs: typing.Any) -> str:
        kwargs_with_defaults: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        return super().json(**kwargs_with_defaults)

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        kwargs_with_defaults_exclude_unset: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        kwargs_with_defaults_exclude_none: typing.Any = {"by_alias": True, "exclude_none": True, **kwargs}

        return deep_union_pydantic_dicts(
            super().dict(**kwargs_with_defaults_exclude_unset), super().dict(**kwargs_with_defaults_exclude_none)
        )

    class Config:
        frozen = True
        smart_union = True
        extra = pydantic_v1.Extra.allow
        json_encoders = {dt.datetime: serialize_datetime}
