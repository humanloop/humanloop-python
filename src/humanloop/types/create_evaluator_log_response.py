# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic

from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.unchecked_base_model import UncheckedBaseModel


class CreateEvaluatorLogResponse(UncheckedBaseModel):
    id: str = pydantic.Field()
    """
    String identifier of the new Log.
    """

    parent_id: str = pydantic.Field()
    """
    Identifier of the evaluated parent Log.
    """

    session_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    Identifier of the Session containing both the parent and the new child Log. If the parent Log does not belong to a Session, a new Session is created with this ID.
    """

    version_id: str = pydantic.Field()
    """
    Identifier of Evaluator Version for which the Log was registered.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow