# This file was auto-generated by Fern from our API Definition.

import typing

from .anthropic_redacted_thinking_content import AnthropicRedactedThinkingContent
from .anthropic_thinking_content import AnthropicThinkingContent

ChatMessageThinkingItem = typing.Union[AnthropicThinkingContent, AnthropicRedactedThinkingContent]
