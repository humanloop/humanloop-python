from typing import Optional

from opentelemetry import baggage
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider

from humanloop.otel.constants import HL_TRACE_METADATA_KEY
from humanloop.otel.helpers import module_is_installed

"""
Humanloop SDK uses the Baggage concept from OTel
to store the Trace metadata. Read more here:
https://opentelemetry.io/docs/concepts/signals/baggage/

The top of the stack contains the Trace information of
the parent Span.

When a Span is created by a decorator, the metadata of
that Span is pushed to the stack so the children can
peek at it and determine its parent in a Flow Trace.

When the parent Span is completed, the context is popped
off the stack.
"""
_BAGGAGE_CONTEXT_STACK: list[Context] = [Context()]


def instrument_provider(provider: TracerProvider):
    """Add Instrumentors to the TracerProvider.

    Instrumentors intercept calls to libraries such as OpenAI client
    and adds metadata to the Spans created by the decorators.
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

    if module_is_installed("groq"):
        from opentelemetry.instrumentation.groq import GroqInstrumentor

        GroqInstrumentor().instrument(tracer_provider=provider)

    # NOTE: ReplicateInstrumentor would require us to bump minimum Python version from 3.8 to 3.9
    # TODO: Do a PR against the open-source ReplicateInstrumentor to support lower Python versions
    # if module_is_installed("replicate"):
    #     from opentelemetry.instrumentation.replicate import ReplicateInstrumentor

    #     ReplicateInstrumentor().instrument(tracer_provider=provider)


def push_trace_context(trace_metadata: dict):
    """Push Trace metadata for a parent Span.

    Expected to be called when the Span is created
    and before the wrapped function is executed.
    Calling a wrapped function may create children
    Spans, which will need to peek at the parent's
    metadata.
    """
    new_context = baggage.set_baggage(
        HL_TRACE_METADATA_KEY,
        trace_metadata,
        _BAGGAGE_CONTEXT_STACK[-1],
    )
    _BAGGAGE_CONTEXT_STACK.append(new_context)


def pop_trace_context():
    """Clear Trace metadata for a parent Span.

    Expected to be called after the wrapped function
    is executed. This allows Spans on the same level
    to peek at their parent Trace metadata.
    """
    _BAGGAGE_CONTEXT_STACK.pop()


def get_trace_parent_metadata() -> Optional[object]:
    """Peek at Trace metadata stack."""

    return baggage.get_baggage(HL_TRACE_METADATA_KEY, _BAGGAGE_CONTEXT_STACK[-1])
