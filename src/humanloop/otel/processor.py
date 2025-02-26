import logging
from typing import TypedDict


from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

from humanloop.context import get_prompt_context, get_trace_id
from humanloop.otel.constants import (
    HUMANLOOP_FILE_KEY,
    HUMANLOOP_FILE_TYPE_KEY,
    HUMANLOOP_LOG_KEY,
    HUMANLOOP_PATH_KEY,
)
from humanloop.otel.helpers import is_llm_provider_call

logger = logging.getLogger("humanloop.sdk")


class CompletableSpan(TypedDict):
    span: ReadableSpan
    complete: bool


class HumanloopSpanProcessor(SimpleSpanProcessor):
    def __init__(self, exporter: SpanExporter) -> None:
        super().__init__(exporter)

    def on_start(self, span: Span, parent_context):
        if is_llm_provider_call(span):
            prompt_context = get_prompt_context()
            if prompt_context:
                path, template = prompt_context.path, prompt_context.template
                span.set_attribute(HUMANLOOP_PATH_KEY, path)
                span.set_attribute(HUMANLOOP_FILE_TYPE_KEY, "prompt")
                if template:
                    span.set_attribute(
                        f"{HUMANLOOP_FILE_KEY}.template",
                        template,
                    )
            else:
                # TODO: handle
                raise ValueError("Provider call outside @prompt context manager")
            trace_id = get_trace_id()
            if trace_id:
                span.set_attribute(f"{HUMANLOOP_LOG_KEY}.trace_parent_id", trace_id)
                print(span)
