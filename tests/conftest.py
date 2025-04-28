from dataclasses import asdict, dataclass
import os
import random
import string
import time
from typing import Callable, Generator
import typing
from unittest.mock import MagicMock

from dotenv import load_dotenv
import pytest
from humanloop.base_client import BaseHumanloop
from humanloop.client import Humanloop
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


@dataclass
class APIKeys:
    openai: str
    humanloop: str


@pytest.fixture(scope="session")
def api_keys() -> APIKeys:
    openai_key = os.getenv("OPENAI_API_KEY")
    humanloop_key = os.getenv("HUMANLOOP_API_KEY")
    for key_name, key_value in [
        ("OPENAI_API_KEY", openai_key),
        ("HUMANLOOP_API_KEY", humanloop_key),
    ]:
        if key_value is None:
            raise ValueError(f"{key_name} is not set in .env file")
    api_keys = APIKeys(
        openai=openai_key,  # type: ignore [arg-type]
        humanloop=humanloop_key,  # type: ignore [arg-type]
    )
    for key, value in asdict(api_keys).items():
        if value is None:
            raise ValueError(f"{key.upper()} key is not set in .env file")
    return api_keys


@pytest.fixture(scope="session")
def humanloop_client(api_keys: APIKeys) -> Humanloop:
    return Humanloop(
        api_key=api_keys.humanloop,
        base_url="https://neostaging.humanloop.ml/v5/",
    )


@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()


def directory_cleanup(directory_id: str, humanloop_client: Humanloop):
    response = humanloop_client.directories.get(directory_id)
    for file in response.files:
        file_id = file.id
        if file.type == "prompt":
            client = humanloop_client.prompts  # type: ignore [assignment]
        elif file.type == "tool":
            client = humanloop_client.tools  # type: ignore [assignment]
        elif file.type == "dataset":
            client = humanloop_client.datasets  # type: ignore [assignment]
        elif file.type == "evaluator":
            client = humanloop_client.evaluators  # type: ignore [assignment]
        elif file.type == "flow":
            client = humanloop_client.flows  # type: ignore [assignment]
        elif file.type == "agent":
            client = humanloop_client.agents  # type: ignore [assignment]
        client.delete(file_id)

    for subdirectory in response.subdirectories:
        directory_cleanup(
            directory_id=subdirectory.id,
            humanloop_client=humanloop_client,
        )

    humanloop_client.directories.delete(id=response.id)


@dataclass
class DirectoryIdentifiers:
    path: str
    id: str


@pytest.fixture()
def test_directory(
    humanloop_client: Humanloop,
) -> Generator[DirectoryIdentifiers, None, None]:
    # Generate a random  alphanumeric directory name to avoid conflicts
    def get_random_string(length: int = 16) -> str:
        return "".join([random.choice(string.ascii_letters + "0123456789") for _ in range(length)])

    directory_path = "SDK_integ_test_" + get_random_string()
    response = humanloop_client.directories.create(path=directory_path)
    assert response.path == directory_path
    try:
        yield DirectoryIdentifiers(
            path=response.path,
            id=response.id,
        )
    finally:
        time.sleep(1)
        directory_cleanup(response.id, humanloop_client)


@pytest.fixture()
def get_test_path(test_directory: DirectoryIdentifiers) -> Callable[[str], str]:
    def generate_path(name: str) -> str:
        return f"{test_directory.path}/{name}"

    return generate_path


# @pytest.fixture(scope="session", autouse=True)
# def cleanup_test_dirs(humanloop_client: Humanloop):
#     def _cleanup_all_test_dirs():
#         dirs = humanloop_client.directories.list()
#         for dir in dirs:
#             if dir.path.startswith("SDK_integ_test_"):
#                 directory_cleanup(
#                     directory_id=dir.id,
#                     humanloop_client=humanloop_client,
#                 )

#     _cleanup_all_test_dirs()
#     yield
#     _cleanup_all_test_dirs()
