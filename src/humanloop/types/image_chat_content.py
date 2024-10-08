# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
import typing
from .image_url import ImageUrl
import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class ImageChatContent(UncheckedBaseModel):
    type: typing.Literal["image_url"] = "image_url"
    image_url: ImageUrl = pydantic.Field()
    """
    The message's image content.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
