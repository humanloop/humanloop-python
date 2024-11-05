from typing import Optional

from mypy.build import TypedDict
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider
from typing_extensions import NotRequired

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

    if module_is_installed("replicate"):
        from opentelemetry.instrumentation.replicate import ReplicateInstrumentor

        ReplicateInstrumentor().instrument(tracer_provider=provider)


class FlowContext(TypedDict):
    trace_id: NotRequired[str]
    trace_parent_id: NotRequired[Optional[int]]
    is_flow_log: NotRequired[bool]


TRACE_FLOW_CONTEXT: dict[int, FlowContext] = {}
