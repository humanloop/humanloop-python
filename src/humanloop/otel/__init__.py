from typing import Optional, TypedDict

from opentelemetry.sdk.trace import TracerProvider
from typing_extensions import NotRequired

from humanloop.otel.helpers import module_is_installed


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

    if module_is_installed("boto3"):
        from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

        BedrockInstrumentor().instrument(tracer_provider=provider)


class FlowContext(TypedDict):
    trace_id: NotRequired[str]
    trace_parent_id: NotRequired[Optional[int]]
    is_flow_log: NotRequired[bool]


TRACE_FLOW_CONTEXT: dict[int, FlowContext] = {}
