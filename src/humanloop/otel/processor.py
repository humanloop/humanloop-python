import logging


from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

from humanloop.context import get_decorator_context, get_evaluation_context, get_trace_id
from humanloop.otel.constants import (
    HUMANLOOP_FILE_KEY,
    HUMANLOOP_FILE_TYPE_KEY,
    HUMANLOOP_LOG_KEY,
    HUMANLOOP_FILE_PATH_KEY,
)
from humanloop.otel.helpers import is_llm_provider_call

logger = logging.getLogger("humanloop.sdk")


class HumanloopSpanProcessor(SimpleSpanProcessor):
    def __init__(self, exporter: SpanExporter) -> None:
        super().__init__(exporter)

    def on_start(self, span: Span, parent_context=...):
        """Called when a Span is started."""
        if is_llm_provider_call(span):
            decorator_context = get_decorator_context()
            if decorator_context and decorator_context.type == "prompt":
                path, template = (
                    decorator_context.path,
                    decorator_context.version["template"],
                )
                span.set_attribute(HUMANLOOP_FILE_PATH_KEY, path)
                span.set_attribute(HUMANLOOP_FILE_TYPE_KEY, "prompt")
                if template:
                    span.set_attribute(
                        f"{HUMANLOOP_FILE_KEY}.template",
                        template,
                    )
            trace_id = get_trace_id()
            if trace_id:
                span.set_attribute(f"{HUMANLOOP_LOG_KEY}.trace_parent_id", trace_id)

    def on_end(self, span: ReadableSpan):
        """Called when a Span finishes recording."""
        if is_llm_provider_call(span):
            decorator_context = get_decorator_context()
            if decorator_context is None or decorator_context.type != "prompt":
                # User made a provider call outside a @prompt context, ignore the span
                return
            evaluation_context = get_evaluation_context()
            if evaluation_context is not None and evaluation_context.path == decorator_context.path:
                # User made a provider call inside a @prompt context, ignore the span
                return
        self.span_exporter.export([span])
