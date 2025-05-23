# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel


class EvaluateeRequest(UncheckedBaseModel):
    """
    Specification of a File version on Humanloop.

    This can be done in a couple of ways:
    - Specifying `version_id` directly.
    - Specifying a File (and optionally an Environment).
        - A File can be specified by either `path` or `file_id`.
        - An Environment can be specified by `environment_id`. If no Environment is specified, the default Environment is used.
    """

    version_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    Unique identifier for the File Version. If provided, none of the other fields should be specified.
    """

    path: typing.Optional[str] = pydantic.Field(default=None)
    """
    Path identifying a File. Provide either this or `file_id` if you want to specify a File.
    """

    file_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    Unique identifier for the File. Provide either this or `path` if you want to specify a File.
    """

    environment: typing.Optional[str] = pydantic.Field(default=None)
    """
    Name of the Environment a Version is deployed to. Only provide this when specifying a File. If not provided (and a File is specified), the default Environment is used.
    """

    batch_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    Unique identifier for the batch of Logs to include in the Evaluation.
    """

    orchestrated: typing.Optional[bool] = pydantic.Field(default=None)
    """
    Whether the Prompt/Tool is orchestrated by Humanloop. Default is `True`. If `False`, a log for the Prompt/Tool should be submitted by the user via the API.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
