# This file was auto-generated by Fern from our API Definition.

import typing

import typing_extensions

from .validation_error import ValidationErrorParams


class HttpValidationErrorParams(typing_extensions.TypedDict):
    detail: typing_extensions.NotRequired[typing.Sequence[ValidationErrorParams]]
