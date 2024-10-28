import os
from typing import Any, Literal, Optional
import typing

import cohere
import pytest

# replicate has no typing stubs
import replicate  # type: ignore
from dotenv import load_dotenv
from groq import Groq
from humanloop.decorators.prompt import prompt
from humanloop.otel.constants import HL_FILE_OT_KEY
from humanloop.otel.helpers import is_humanloop_span, read_from_opentelemetry_span
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from opentelemetry.sdk.trace import Tracer
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def _call_llm_base(provider: Literal["openai", "anthropic"], messages: list[dict]) -> Optional[str]:
    load_dotenv()
    if provider == "openai":
        # NOTE: These tests check if instrumentors are capable of intercepting OpenAI
        # provider calls. Could not find a way to intercept them coming from a Mock.
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return (
            client.chat.completions.create(
                model="gpt-4o",
                messages=messages,  # type: ignore
                temperature=0.8,
            )
            .choices[0]
            .message.content
        )
    # if provider == "anthropic":
    #     client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    #     return client.messages.create(
    #         model="claude-3-opus",
    #         messages=messages,
    #         max_tokens=200,
    #     ).content
    # if provider == "mistralai":
    #     client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
    #     response = client.chat(model="mistral-small-latest", messages=messages, temperature=0.8)
    #     return response.choices[0].message.content
    if provider == "groq":
        # Note GROQ might be unavailable, leading to
        # test failure. Returns groq.NotFoundError: Not Found
        client = Groq(
            # This is the default and can be omitted
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        return (
            client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",
                temperature=0.8,
            )
            .choices[0]
            .message.content
        )
    if provider == "cohere":
        client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        messages_cohere_format = []
        for message in messages:
            if message["role"] == "system":
                messages_cohere_format.append(cohere.SystemMessage(message=message["content"]))
            elif message["role"] == "user":
                messages_cohere_format.append(cohere.UserMessage(message=message["content"]))
            elif message["role"] == "assistant":
                messages_cohere_format.append(cohere.ChatbotMessage(message=message["content"]))
        return client.chat(
            chat_history=messages_cohere_format,
            model="command",
            max_tokens=200,
            message=messages[-1]["content"],
            temperature=0.8,
        ).text
    if provider == "replicate":
        # TODO: Instrumentor only picks up methods on module-level, not client level
        replicate.default_client._api_token = os.getenv("REPLICATE_API_KEY")
        output = ""
        for event in replicate.run(
            "meta/meta-llama-3-8b-instruct",
            input={
                "prompt": messages[0]["content"] + " " + messages[-1]["content"],
                "temperature": 0.8,
            },
        ):
            output += str(event)
        return output
    raise ValueError(f"Unknown provider: {provider}")


# prompt is a decorator, but for sake of brevity, I am using it as a higher-order function
_call_llm = prompt(
    path=None,
    template="You are an assistant on the following topics: {topics}.",
)(_call_llm_base)
_call_llm_with_defaults = prompt(
    path=None,
    template="You are an assistant on the following topics: {topics}.",
    temperature=0.9,
    top_p=0.1,
)(_call_llm_base)


@pytest.mark.parametrize(
    "provider",
    (
        "openai",
        "groq",
        "cohere",
        "replicate",
    ),
)
def test_prompt(
    provider: str,
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
    call_llm_messages: list[ChatCompletionMessageParam],
):
    # GIVEN a default OpenTelemetry configuration
    _, exporter = opentelemetry_test_configuration
    # WHEN using the Prompt decorator
    _call_llm(provider=provider, messages=call_llm_messages)
    # THEN two spans are created: one for the OpenAI LLM provider call and one for the Prompt
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    # THEN the Prompt span is not enhanced with information from the LLM provider
    assert is_humanloop_span(spans[1])
    assert spans[1].attributes.get("prompt") is None  # type: ignore


@pytest.mark.parametrize(
    "provider",
    (
        "openai",
        "groq",
        "cohere",
        "replicate",
    ),
)
def test_prompt_hl_processor(
    provider: str,
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
    call_llm_messages: list[ChatCompletionMessageParam],
):
    # GIVEN an OpenTelemetry configuration with a Humanloop Span processor
    _, exporter = opentelemetry_hl_test_configuration
    # WHEN using the Prompt decorator
    _call_llm(provider=provider, messages=call_llm_messages)
    # THEN a single span is created since the LLM provider call span is merged in the Prompt span
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert is_humanloop_span(span=spans[0])
    prompt: dict[str, Any] = read_from_opentelemetry_span(span=spans[0], key=HL_FILE_OT_KEY)["prompt"]  # type: ignore
    assert prompt is not None
    # THEN temperature is taken from LLM provider call, but top_p is not since it is not specified
    assert prompt["temperature"] == 0.8
    assert prompt["provider"] == provider
    assert prompt.get("top_p") is None


@pytest.mark.parametrize(
    "provider",
    (
        "openai",
        "groq",
        "cohere",
        "replicate",
    ),
)
def test_prompt_with_defaults(
    provider: str,
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
    call_llm_messages: list[ChatCompletionMessageParam],
):
    # GIVEN an OpenTelemetry configuration with a Humanloop Span processor
    _, exporter = opentelemetry_hl_test_configuration
    # WHEN using the Prompt decorator with default values
    _call_llm_with_defaults(provider=provider, messages=call_llm_messages)
    # THEN a single span is created since the LLM provider call span is merged in the Prompt span
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert is_humanloop_span(spans[0])
    prompt: dict[str, Any] = read_from_opentelemetry_span(span=spans[0], key=HL_FILE_OT_KEY)["prompt"]  # type: ignore
    assert prompt is not None
    # THEN temperature is taken from decorator rather than intercepted LLM provider call
    assert prompt["temperature"] == 0.9
    # THEN top_p is present
    assert prompt["top_p"] == 0.1


@pytest.mark.parametrize(
    "hyperparameters",
    (
        {"temperature": 1.1},
        {"top_p": 1.1},
        {"presence_penalty": 3},
        {"frequency_penalty": 3},
    ),
)
def test_default_values_fails_out_of_domain(hyperparameters: dict[str, float]):
    # GIVEN a Prompt decorated function
    # WHEN using default values that are out of domain
    # THEN an exception is raised
    with pytest.raises(ValueError):

        @prompt(path=None, template="You are an assistant on the following topics: {topics}.", **hyperparameters)  # type: ignore[arg-type]
        def _call_llm(messages: list[ChatCompletionMessageParam]) -> Optional[str]:
            load_dotenv()
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            return (
                client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.8,
                )
                .choices[0]
                .message.content
            )
