import json
import logging
from collections import defaultdict
from typing import Any

# No typing stubs for parse
import parse  # type: ignore
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

from humanloop.otel.constants import HL_FILE_OT_KEY, HL_LOG_OT_KEY
from humanloop.otel.helpers import (
    is_humanloop_span,
    is_llm_provider_call,
    read_from_opentelemetry_span,
    write_to_opentelemetry_span,
)


class HumanloopSpanProcessor(SimpleSpanProcessor):
    """Merge information from Instrumentors used by Humanloop SDK into the
    Spans that will be exported to Humanloop.
    """

    def __init__(self, exporter: SpanExporter) -> None:
        super().__init__(exporter)
        # Span parent to Span children map
        self._children: dict[int, list] = defaultdict(list)

    # TODO: Could override on_start and process Flow spans ahead of time
    # and PATCH the created Logs in on_end. A special type of ReadableSpan could be
    # used for this

    def on_end(self, span: ReadableSpan) -> None:
        if is_humanloop_span(span=span):
            _process_humanloop_span(span, self._children[span.context.span_id])
            del self._children[span.context.span_id]
            self.span_exporter.export([span])
        else:
            if span.parent is not None and _is_instrumentor_span(span):
                self._children[span.parent.span_id].append(span)


def _is_instrumentor_span(span: ReadableSpan) -> bool:
    # TODO: Extend in the future as needed. Spans not coming from
    # Instrumentors of interest should be dropped
    return is_llm_provider_call(span=span)


def _process_humanloop_span(span: ReadableSpan, children_spans: list[ReadableSpan]):
    hl_file = read_from_opentelemetry_span(span, key=HL_FILE_OT_KEY)

    if "prompt" in hl_file:
        _process_prompt(prompt_span=span, children_spans=children_spans)
        return
    elif "tool" in hl_file:
        _process_tool(tool_span=span, children_spans=children_spans)
        return
    elif "flow" in hl_file:
        _process_flow(flow_span=span, children_spans=children_spans)
        return
    else:
        logging.error("Invalid span type")


def _process_prompt(prompt_span: ReadableSpan, children_spans: list[ReadableSpan]):
    if len(children_spans) == 0:
        return
    child_span = children_spans[0]
    assert is_llm_provider_call(child_span)
    _enrich_prompt_span_file(prompt_span, child_span)
    _enrich_prompt_span_log(prompt_span, child_span)


def _process_tool(tool_span: ReadableSpan, children_spans: list[ReadableSpan]):
    # TODO: Use children_spans in the future
    tool_log = read_from_opentelemetry_span(tool_span, key=HL_LOG_OT_KEY)
    if tool_span.start_time:
        tool_log["start_time"] = tool_span.start_time / 1e9
    if tool_span.end_time:
        tool_log["end_time"] = tool_span.end_time / 1e9
        tool_log["created_at"] = tool_span.end_time / 1e9

    write_to_opentelemetry_span(
        span=tool_span,
        key=HL_LOG_OT_KEY,
        value=tool_log,
    )


def _process_flow(flow_span: ReadableSpan, children_spans: list[ReadableSpan]):
    # TODO: Use children_spans in the future
    flow_log = read_from_opentelemetry_span(flow_span, key=HL_LOG_OT_KEY)
    if flow_span.start_time:
        flow_log["start_time"] = flow_span.start_time / 1e9
    if flow_span.end_time:
        flow_log["end_time"] = flow_span.end_time / 1e9
        flow_log["created_at"] = flow_span.end_time / 1e9

    write_to_opentelemetry_span(
        span=flow_span,
        key=HL_LOG_OT_KEY,
        value=flow_log,
    )


def _enrich_prompt_span_file(prompt_span: ReadableSpan, llm_provider_call_span: ReadableSpan):
    hl_file: dict[str, Any] = read_from_opentelemetry_span(prompt_span, key=HL_FILE_OT_KEY)
    gen_ai_object: dict[str, Any] = read_from_opentelemetry_span(llm_provider_call_span, key="gen_ai")
    llm_object: dict[str, Any] = read_from_opentelemetry_span(llm_provider_call_span, key="llm")

    prompt_kernel: dict[str, Any] = hl_file.get("prompt", {})
    if "model" not in prompt_kernel:
        prompt_kernel["model"] = gen_ai_object.get("request", {}).get("model", None)
    if "endpoint" not in prompt_kernel:
        prompt_kernel["endpoint"] = llm_object.get("request", {}).get("type")
    if "template" not in prompt_kernel:
        prompt_kernel["template"] = hl_file.get("prompt", {}).get("template", None)
    if "provider" not in prompt_kernel:
        prompt_kernel["provider"] = gen_ai_object.get("system", None)
        if prompt_kernel["provider"]:
            prompt_kernel["provider"] = prompt_kernel["provider"].lower()
            if prompt_kernel["provider"] not in [
                "openai",
                "openai_azure",
                "mock",
                "anthropic",
                "bedrock",
                "cohere",
                "replicate",
                "google",
                "groq",
            ]:
                raise ValueError("Invalid provider")
    if "temperature" not in prompt_kernel:
        prompt_kernel["temperature"] = gen_ai_object.get("request", {}).get("temperature", None)
    if "top_p" not in prompt_kernel:
        prompt_kernel["top_p"] = gen_ai_object.get("request", {}).get("top_p", None)
    if "max_tokens" not in prompt_kernel:
        prompt_kernel["max_tokens"] = gen_ai_object.get("request", {}).get("max_tokens", None)
    if "presence_penalty" not in prompt_kernel:
        prompt_kernel["presence_penalty"] = llm_object.get("presence_penalty", None)
    if "frequency_penalty" not in prompt_kernel:
        prompt_kernel["frequency_penalty"] = llm_object.get("frequency_penalty", None)

    write_to_opentelemetry_span(
        span=prompt_span,
        key=HL_FILE_OT_KEY,
        # hl_file was modified in place via prompt_kernel reference
        value=hl_file,
    )


def _enrich_prompt_span_log(prompt_span: ReadableSpan, llm_provider_call_span: ReadableSpan):
    hl_file: dict[str, Any] = read_from_opentelemetry_span(prompt_span, key=HL_FILE_OT_KEY)
    hl_log: dict[str, Any] = read_from_opentelemetry_span(prompt_span, key=HL_LOG_OT_KEY)
    gen_ai_object: dict[str, Any] = read_from_opentelemetry_span(llm_provider_call_span, key="gen_ai")

    # TODO: Seed not added by Instrumentors in provider call

    if "output_tokens" not in hl_log:
        hl_log["output_tokens"] = gen_ai_object.get("usage", {}).get("completion_tokens")
    if len(gen_ai_object.get("completion", [])) > 0:
        hl_log["finish_reason"] = gen_ai_object.get("completion", {}).get("0", {}).get("finish_reason")
    hl_log["messages"] = gen_ai_object.get("prompt", [])

    if prompt_span.start_time:
        hl_log["start_time"] = prompt_span.start_time / 1e9
    if prompt_span.end_time:
        hl_log["end_time"] = prompt_span.end_time / 1e9
        hl_log["created_at"] = prompt_span.end_time / 1e9

    try:
        inputs = {}
        system_message = gen_ai_object["prompt"]["0"]["content"]
        template = hl_file["prompt"]["template"]
        parsed = parse.parse(template, system_message)
        for key, value in parsed.named.items():
            try:
                parsed_value = json.loads(value.replace("'", '"'))
            except json.JSONDecodeError:
                parsed_value = value
            inputs[key] = parsed_value
    except Exception as e:
        logging.error(e)
        inputs = {}
    finally:
        hl_log["inputs"] = inputs

    write_to_opentelemetry_span(
        span=prompt_span,
        key=HL_LOG_OT_KEY,
        # hl_log was modified in place
        value=hl_log,
    )
