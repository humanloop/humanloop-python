from typing import Optional

from opentelemetry import baggage
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer

from humanloop.otel.constants import HL_TRACE_METADATA_KEY
from humanloop.otel.helpers import module_is_installed

_TRACER = None

_BAGGAGE_CONTEXT: list[Context] = [Context()]


def set_tracer(tracer: Tracer):
    global _TRACER
    _TRACER = tracer


def get_tracer() -> Tracer:
    assert _TRACER is not None, "Internal error: OTT Tracer should have been set in the client"
    return _TRACER


def instrument_provider(provider: TracerProvider):
    """Add Instrumentors to the TracerProvider.

    Instrumentors add extra spans which are merged in Humanloop Span logs.
    """
    if module_is_installed("openai"):
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument(tracer_provider=provider)

    if module_is_installed("cohere"):
        from opentelemetry.instrumentation.cohere import CohereInstrumentor

        CohereInstrumentor().instrument(tracer_provider=provider)

    if module_is_installed("anthropic"):
        from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

        AnthropicInstrumentor().instrument(tracer_provider=provider)

    if module_is_installed("mistralai"):
        from opentelemetry.instrumentation.mistralai import MistralAiInstrumentor

        # TODO: Need to to a PR to the instrumentor to support > 1.0.0 Mistral clients
        MistralAiInstrumentor().instrument(tracer_provider=provider)

    if module_is_installed("groq"):
        from opentelemetry.instrumentation.groq import GroqInstrumentor

        GroqInstrumentor().instrument(tracer_provider=provider)

    if module_is_installed("replicate"):
        from opentelemetry.instrumentation.replicate import ReplicateInstrumentor

        ReplicateInstrumentor().instrument(tracer_provider=provider)


def push_trace_context(trace_metadata: dict):
    """Set metadata for Trace parent.

    Used before the wrapped function is executed. All decorated functions
    called from the decorated function will use this metadata to determine
    the Log it should be associated to in Flow Trace.
    """
    global _BAGGAGE_CONTEXT
    new_context = baggage.set_baggage(
        HL_TRACE_METADATA_KEY,
        trace_metadata,
        _BAGGAGE_CONTEXT[-1],
    )
    _BAGGAGE_CONTEXT.append(new_context)


def pop_trace_context():
    """Clear Trace parent metadata.

    Used after the wrapped function has been executed.
    """
    global _BAGGAGE_CONTEXT
    _BAGGAGE_CONTEXT.pop()


def get_trace_context() -> Optional[object]:
    """Get Trace parent metadata for Flows."""

    global _BAGGAGE_CONTEXT

    return baggage.get_baggage(HL_TRACE_METADATA_KEY, _BAGGAGE_CONTEXT[-1])
