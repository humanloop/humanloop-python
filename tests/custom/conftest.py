from typing import Generator
import typing
from unittest.mock import MagicMock

import pytest
from humanloop.base_client import BaseHumanloop
from humanloop.otel.exporter import HumanloopSpanExporter
from humanloop.otel.processor import HumanloopSpanProcessor
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.cohere import CohereInstrumentor
from opentelemetry.instrumentation.groq import GroqInstrumentor
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.replicate import ReplicateInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

if typing.TYPE_CHECKING:
    from humanloop.client import BaseHumanloop


@pytest.fixture(scope="function")
def opentelemetry_test_provider() -> TracerProvider:
    """Create a test TracerProvider with a resource.

    This is similar to the created TracerProvider in the
    Humanloop class.
    """
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service": "humanloop.sdk",
                "environment": "test",
            }
        )
    )
    return provider


@pytest.fixture(scope="function")
def test_span(opentelemetry_test_provider: TracerProvider):
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    opentelemetry_test_provider.add_span_processor(processor)
    tracer = opentelemetry_test_provider.get_tracer("test")
    return tracer.start_span("test_span")


@pytest.fixture(scope="function")
def opentelemetry_test_configuration(
    opentelemetry_test_provider: TracerProvider,
) -> Generator[tuple[Tracer, InMemorySpanExporter], None, None]:
    """Configure OTel backend without HumanloopSpanProcessor.

    Spans created by Instrumentors will not be used to enrich
    Humanloop Spans.
    """
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    opentelemetry_test_provider.add_span_processor(processor)
    instrumentors: list[BaseInstrumentor] = [
        OpenAIInstrumentor(),
        AnthropicInstrumentor(),
        GroqInstrumentor(),
        CohereInstrumentor(),
        ReplicateInstrumentor(),
    ]
    for instrumentor in instrumentors:
        instrumentor.instrument(tracer_provider=opentelemetry_test_provider)
    tracer = opentelemetry_test_provider.get_tracer("test")
    # Circumvent configuration procedure

    yield tracer, exporter

    for instrumentor in instrumentors:
        instrumentor.uninstrument()


@pytest.fixture(scope="function")
def opentelemetry_hl_test_configuration(
    opentelemetry_test_provider: TracerProvider,
    humanloop_client: BaseHumanloop,
) -> Generator[tuple[Tracer, InMemorySpanExporter], None, None]:
    """Configure OTel backend with HumanloopSpanProcessor.

    Spans created by Instrumentors will be used to enrich
    Humanloop Spans.
    """
    exporter = InMemorySpanExporter()
    processor = HumanloopSpanProcessor(exporter=exporter)
    opentelemetry_test_provider.add_span_processor(processor)
    instrumentors: list[BaseInstrumentor] = [
        OpenAIInstrumentor(),
        AnthropicInstrumentor(),
        GroqInstrumentor(),
        CohereInstrumentor(),
        ReplicateInstrumentor(),
        AnthropicInstrumentor(),
    ]
    for instrumentor in instrumentors:
        instrumentor.instrument(
            tracer_provider=opentelemetry_test_provider,
        )
    tracer = opentelemetry_test_provider.get_tracer("test")

    yield tracer, exporter

    for instrumentor in instrumentors:
        instrumentor.uninstrument()


@pytest.fixture(scope="function")
def hl_test_exporter() -> HumanloopSpanExporter:
    """
    Test Exporter where HTTP calls to Humanloop API
    are mocked.
    """
    client = MagicMock()
    exporter = HumanloopSpanExporter(client=client)
    return exporter


@pytest.fixture(scope="function")
def opentelemetry_hl_with_exporter_test_configuration(
    hl_test_exporter: HumanloopSpanExporter,
    opentelemetry_test_provider: TracerProvider,
) -> Generator[tuple[Tracer, HumanloopSpanExporter], None, None]:
    """Configure OTel backend with HumanloopSpanProcessor and
    a HumanloopSpanExporter where HTTP calls are mocked.
    """
    processor = HumanloopSpanProcessor(exporter=hl_test_exporter)
    opentelemetry_test_provider.add_span_processor(processor)
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(tracer_provider=opentelemetry_test_provider)
    tracer = opentelemetry_test_provider.get_tracer("test")

    yield tracer, hl_test_exporter

    instrumentor.uninstrument()


@pytest.fixture(scope="session")
def call_llm_messages() -> list[ChatCompletionMessageParam]:
    return [
        {
            "role": "system",
            "content": "You are an assistant on the following topics: greetings in foreign languages.",
        },
        {
            "role": "user",
            "content": "Bonjour!",
        },
    ]
