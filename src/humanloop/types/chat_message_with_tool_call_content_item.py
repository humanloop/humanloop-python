# This file was auto-generated by Fern from our API Definition.

import typing

from .image_chat_content import ImageChatContent
from .text_chat_content import TextChatContent

ChatMessageWithToolCallContentItem = typing.Union[TextChatContent, ImageChatContent]
