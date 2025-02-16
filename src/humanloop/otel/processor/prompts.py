import logging
from typing import Any
from opentelemetry.sdk.trace import ReadableSpan
from pydantic import ValidationError as PydanticValidationError

from humanloop.otel.constants import HUMANLOOP_FILE_KEY, HUMANLOOP_LOG_KEY
from humanloop.otel.helpers import (
    is_llm_provider_call,
    read_from_opentelemetry_span,
    write_to_opentelemetry_span,
)
from humanloop.types.prompt_kernel_request import PromptKernelRequest

logger = logging.getLogger("humanloop.sdk")


def enhance_prompt_span(prompt_span: ReadableSpan, dependencies: list[ReadableSpan]):
    """Add information from the LLM provider span to the Prompt span.

    We are passing a list of children spans to the Prompt span, but more than one
    is undefined behavior.
    """
    if len(dependencies) == 0:
        return
    for child_span in dependencies:
        if is_llm_provider_call(child_span):
            _enrich_prompt_kernel(prompt_span, child_span)
            _enrich_prompt_log(prompt_span, child_span)
            # NOTE: @prompt decorator expects a single LLM provider call
            # to happen in the function. If there are more than one, we
            # ignore the rest
            break


def _enrich_prompt_kernel(prompt_span: ReadableSpan, llm_provider_call_span: ReadableSpan):
    hl_file: dict[str, Any] = read_from_opentelemetry_span(prompt_span, key=HUMANLOOP_FILE_KEY)
    gen_ai_object: dict[str, Any] = read_from_opentelemetry_span(llm_provider_call_span, key="gen_ai")
    llm_object: dict[str, Any] = read_from_opentelemetry_span(llm_provider_call_span, key="llm")

    prompt: dict[str, Any] = hl_file.get("prompt", {})  # type: ignore

    # Check if the Prompt Kernel keys were assigned default values
    # via the @prompt arguments. Otherwise, use the information
    # from the intercepted LLM provider call
    prompt["model"] = prompt.get("model") or gen_ai_object.get("request", {}).get("model", None)
    if prompt["model"] is None:
        raise ValueError("Could not infer required parameter `model`. Please provide it in the @prompt decorator.")
    prompt["endpoint"] = prompt.get("endpoint") or llm_object.get("request", {}).get("type")
    prompt["provider"] = prompt.get("provider") or gen_ai_object.get("system", None)
    if prompt["provider"]:
        # Normalize provider name; Interceptors output the names with
        # different capitalization e.g. OpenAI instead of openai
        prompt["provider"] = prompt["provider"].lower()
    prompt["temperature"] = prompt.get("temperature") or gen_ai_object.get("request", {}).get("temperature", None)
    prompt["top_p"] = prompt.get("top_p") or gen_ai_object.get("request", {}).get("top_p", None)
    prompt["max_tokens"] = prompt.get("max_tokens") or gen_ai_object.get("request", {}).get("max_tokens", None)
    prompt["presence_penalty"] = prompt.get("presence_penalty") or llm_object.get("presence_penalty", None)
    prompt["frequency_penalty"] = prompt.get("frequency_penalty") or llm_object.get("frequency_penalty", None)
    prompt["tools"] = prompt.get("tools", [])

    try:
        # Validate the Prompt Kernel
        PromptKernelRequest.model_validate(obj=prompt)  # type: ignore
    except PydanticValidationError as e:
        logger.error(
            "[HumanloopSpanProcessor] Could not validate Prompt Kernel extracted from span: %s %s. Error: %s",
            prompt_span.context.span_id,
            prompt_span.name,
            e,
        )

    # Write the enriched Prompt Kernel back to the span
    hl_file["prompt"] = prompt
    write_to_opentelemetry_span(
        span=prompt_span,
        key=HUMANLOOP_FILE_KEY,
        # hl_file was modified in place via prompt_kernel reference
        value=hl_file,
    )


def _enrich_prompt_log(prompt_span: ReadableSpan, llm_provider_call_span: ReadableSpan):
    try:
        hl_log: dict[str, Any] = read_from_opentelemetry_span(prompt_span, key=HUMANLOOP_LOG_KEY)
    except KeyError:
        hl_log = {}
    gen_ai_object: dict[str, Any] = read_from_opentelemetry_span(llm_provider_call_span, key="gen_ai")

    # TODO: Seed not added by Instrumentors in provider call

    if "output_tokens" not in hl_log:
        hl_log["output_tokens"] = gen_ai_object.get("usage", {}).get("completion_tokens")
    if len(gen_ai_object.get("completion", [])) > 0:
        hl_log["finish_reason"] = gen_ai_object["completion"][0].get("finish_reason")
    hl_log["messages"] = gen_ai_object.get("prompt")

    write_to_opentelemetry_span(
        span=prompt_span,
        key=HUMANLOOP_LOG_KEY,
        # hl_log was modified in place
        value=hl_log,
    )
