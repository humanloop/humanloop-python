# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions

from .validation_error_loc_item import ValidationErrorLocItemParams


class ValidationErrorParams(typing_extensions.TypedDict):
    loc: typing.Sequence[ValidationErrorLocItemParams]
    msg: str
    type: str
