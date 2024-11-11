import logging
from collections import defaultdict
from typing import Any

# No typing stubs for parse
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from pydantic import ValidationError as PydanticValidationError

from humanloop.otel.constants import HUMANLOOP_FILE_KEY, HUMANLOOP_FILE_TYPE_KEY, HUMANLOOP_LOG_KEY
from humanloop.otel.helpers import (
    is_humanloop_span,
    is_llm_provider_call,
    read_from_opentelemetry_span,
    write_to_opentelemetry_span,
)
from humanloop.types.prompt_kernel_request import PromptKernelRequest

logger = logging.getLogger("humanloop.sdk")


class HumanloopSpanProcessor(SimpleSpanProcessor):
    """Enrich Humanloop spans with data from their children spans.

    The decorators add Instrumentors to the OpenTelemetry TracerProvider
    that log interactions with common LLM libraries. These Instrumentors
    produce Spans which contain information that can be used to enrich the
    Humanloop File Kernels.

    For example, Instrumentors for LLM provider libraries intercept
    hyperparameters used in the API call to the model to build the
    Prompt File definition when using the @prompt decorator.

    Spans created that are not created by Humanloop decorators, such as
    those created by the Instrumentors mentioned above, will be passed
    to the Exporter as they are.
    """

    def __init__(self, exporter: SpanExporter) -> None:
        super().__init__(exporter)
        # Span parent to Span children map
        self._children: dict[int, list] = defaultdict(list)

    # NOTE: Could override on_start and process Flow spans ahead of time
    # and PATCH the created Logs in on_end. A special type of ReadableSpan could be
    # used for this

    def on_end(self, span: ReadableSpan) -> None:
        if is_humanloop_span(span=span):
            _process_span_dispatch(span, self._children[span.context.span_id])
            # Release the reference to the Spans as they've already
            # been sent to the Exporter
            del self._children[span.context.span_id]
        else:
            if span.parent is not None and _is_instrumentor_span(span):
                # Copy the Span and keep it until the Humanloop Span
                # arrives in order to enrich it
                self._children[span.parent.span_id].append(span)
        # Pass the Span to the Exporter
        self.span_exporter.export([span])


def _is_instrumentor_span(span: ReadableSpan) -> bool:
    """Determine if the Span contains information of interest for Spans created by Humanloop decorators."""
    # At the moment we only enrich Spans created by the Prompt decorators
    # As we add Instrumentors for other libraries, this function must
    # be expanded
    return is_llm_provider_call(span=span)


def _process_span_dispatch(span: ReadableSpan, children_spans: list[ReadableSpan]):
    file_type = span.attributes[HUMANLOOP_FILE_TYPE_KEY]  # type: ignore

    # Processing common to all Humanloop File types
    if span.start_time:
        span._attributes[f"{HUMANLOOP_LOG_KEY}.start_time"] = int(span.start_time / 1e9)  # type: ignore
    if span.end_time:
        span._attributes[f"{HUMANLOOP_LOG_KEY}.end_time"] = int(span.end_time / 1e9)  # type: ignore
        span._attributes[f"{HUMANLOOP_LOG_KEY}.created_at"] = int(span.end_time / 1e9)  # type: ignore

    # Processing specific to each Humanloop File type
    if file_type == "prompt":
        _process_prompt(prompt_span=span, children_spans=children_spans)
        return
    elif file_type == "tool":
        pass
    elif file_type == "flow":
        pass
    else:
        logger.error("Unknown Humanloop File Span %s", span)


def _process_prompt(prompt_span: ReadableSpan, children_spans: list[ReadableSpan]):
    if len(children_spans) == 0:
        return
    for child_span in children_spans:
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
        PromptKernelRequest.model_validate(obj=prompt)
    except PydanticValidationError as e:
        logger.error("Could not validate Prompt Kernel extracted from Span: %s", e)

    # Write the enriched Prompt Kernel back to the Span
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
