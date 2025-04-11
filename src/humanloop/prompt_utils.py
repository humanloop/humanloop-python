import copy
from typing import Any, Dict, List, Optional, TypeVar, Sequence
import logging

import re

from .requests.chat_message import ChatMessageParams
from .prompts.requests.prompt_request_template import (
    PromptRequestTemplateParams,
)


logger = logging.getLogger(__name__)


class PromptVariablesNotFoundError(ValueError):
    """Raised when inputs do not satisfy prompt variables."""

    missing_variables: List[str]
    """Missing variables"""

    def __init__(self, missing_variables: List[str]) -> None:
        self.missing_variables = missing_variables
        super().__init__(f"Prompt requires inputs for the following " f"variables: {self.missing_variables}")


def populate_prompt_template(
    template: str,
    inputs: Optional[Dict[str, Any]],
) -> str:
    """Interpolate a string template with kwargs, where template variables
    are specified using double curly bracket syntax: {{variable}}.

    args:
        template: str - string template where template variables are specified
        using double curly bracket syntax: {{variable}}.

        inputs - represent the key, value string pairs to inject into the template
        variables, where key corresponds to the template variable name and
        value to the variable value to inject

    return:
        The interpolated template string

    raises:
        PromptVariablesNotFoundError - if any variables are missing from inputs
    """
    template_variables: List[str] = re.findall(
        # Matching variables: `{{ variable_2 }}`
        r"{{\s?([a-zA-Z_\d\.\[\]]+)\s?}}",
        template,
    ) + re.findall(
        # Matching tools: `{{ tool_2("all characters$#@$!") }}`
        # https://regexr.com/7nvrf
        r"\{\{\s?([a-zA-Z_\-\d]+\([a-zA-Z_\-\d,\s\"]+\))\s?\}\}",
        template,
    )

    # populate the template variables, tracking if any are missing
    prompt = template
    missing_vars = []

    if inputs is None:
        inputs = {}

    # e.g. var: input_name, sig(input_name), sig(other_name), sig("string")
    for var in template_variables:
        text: Optional[str] = None

        if var in inputs:
            text = inputs[var]

        if text is None:
            missing_vars.append(var)
        else:
            if not isinstance(text, str):
                logger.info(f"Converting input value for variable '{var}' to string for prompt template: " f"{text}")
                text = str(text)
            replacement = sanitize_prompt(prompt=text) if text else text
            prompt = re.sub(
                r"{{\s?" + re.escape(var) + r"\s?}}",
                replacement,
                prompt,
            )

    if missing_vars:
        missing_vars.sort()
        raise PromptVariablesNotFoundError(
            missing_variables=missing_vars,
        )

    return prompt


def sanitize_prompt(prompt: str):
    return prompt.replace("\\", "\\\\")


def populate_chat_template(
    chat_template: Sequence[ChatMessageParams],
    inputs: Optional[Dict[str, str]] = None,
) -> List[ChatMessageParams]:
    """Interpolate a chat template with kwargs, where template variables."""
    messages = []
    message: ChatMessageParams
    for message in chat_template:
        if "content" not in message:
            messages.append(message)
            continue

        message_content = copy.deepcopy(message["content"])
        if isinstance(message_content, str):
            message_content = populate_prompt_template(
                template=message_content,
                inputs=inputs,
            )
        elif isinstance(message_content, list):
            for j, content_item in enumerate(message_content):
                if content_item["type"] == "text":
                    content_item_text = content_item["text"]
                    (content_item_text,) = populate_prompt_template(
                        template=content_item_text,
                        inputs=inputs,
                    )
                    content_item["text"] = content_item_text
        messages.append(
            ChatMessageParams(
                role=message["role"],
                content=message_content,
            )
        )
    return messages


T = TypeVar("T", bound=PromptRequestTemplateParams)


def populate_template(template: T, inputs: Dict[str, str]) -> T:
    """Populate a Prompt's template with the given inputs.

    Humanloop supports insertion of variables of the form `{{variable}}` in
    Prompt templates.
    E.g. If you provide the template `Hello {{name}}` and the input
    `{"name": "Alice"}`, the populated template will be `Hello Alice`.

    This function supports both completion and chat models. For completion
    models, provide template as a string. For chat models, provide template
    as a list of messages.
    """
    if isinstance(template, str):
        return populate_prompt_template(
            template=template,
            inputs=inputs,
        )
    return populate_chat_template(
        chat_template=template,
        inputs=inputs,
    )
