import logging
from typing import TypedDict


from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

from humanloop.context import get_prompt_path
from humanloop.otel.constants import (
    HUMANLOOP_PATH_KEY,
)
from humanloop.otel.helpers import is_llm_provider_call

logger = logging.getLogger("humanloop.sdk")


class CompletableSpan(TypedDict):
    span: ReadableSpan
    complete: bool


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

    def on_start(self, span: Span):
        if is_llm_provider_call(span):
            prompt_path = get_prompt_path()
            if prompt_path:
                span.set_attribute(HUMANLOOP_PATH_KEY, prompt_path)
            else:
                raise ValueError("Provider call outside @prompt context manager")
