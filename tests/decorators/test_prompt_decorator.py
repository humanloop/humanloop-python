import os
from typing import Any, Optional

import pytest
from dotenv import load_dotenv
from humanloop.decorators.prompt import prompt
from humanloop.otel.constants import HL_FILE_OT_KEY
from humanloop.otel.helpers import NestedDict, is_humanloop_span, read_from_opentelemetry_span
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from opentelemetry.sdk.trace import Tracer
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@prompt(path=None, template="You are an assistant on the following topics: {topics}.")
def _call_llm(messages: list[ChatCompletionMessageParam]) -> Optional[str]:
    load_dotenv()
    # NOTE: These tests check if instrumentors are capable of intercepting OpenAI
    # provider calls. Could not find a way to intercept them coming from a Mock.
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


@prompt(path=None, template="You are an assistant on the following topics: {topics}.", temperature=0.9, top_p=0.1)
def _call_llm_with_defaults(messages: list[ChatCompletionMessageParam]) -> Optional[str]:
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


def test_prompt(
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
    call_llm_messages: list[ChatCompletionMessageParam],
):
    # GIVEN a default OpenTelemetry configuration
    _, exporter = opentelemetry_test_configuration
    # WHEN using the Prompt decorator
    _call_llm(messages=call_llm_messages)
    # THEN two spans are created: one for the OpenAI LLM provider call and one for the Prompt
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    # THEN the Prompt span is not enhanced with information from the LLM provider
    assert is_humanloop_span(spans[1])
    assert spans[1].attributes.get("prompt") is None  # type: ignore


def test_prompt_hl_processor(
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
    call_llm_messages: list[ChatCompletionMessageParam],
):
    # GIVEN an OpenTelemetry configuration with a Humanloop Span processor
    _, exporter = opentelemetry_hl_test_configuration
    # WHEN using the Prompt decorator
    _call_llm(messages=call_llm_messages)
    # THEN a single span is created since the LLM provider call span is merged in the Prompt span
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert is_humanloop_span(spans[0])
    prompt: dict[str, Any] = read_from_opentelemetry_span(span=spans[0], key=HL_FILE_OT_KEY)["prompt"]  # type: ignore
    assert prompt is not None
    # THEN temperature is taken from LLM provider call, but top_p is not since it is not specified
    assert prompt["temperature"] == 0.8
    assert prompt.get("top_p") is None


def test_prompt_with_defaults(
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
    call_llm_messages: list[ChatCompletionMessageParam],
):
    # GIVEN an OpenTelemetry configuration with a Humanloop Span processor
    _, exporter = opentelemetry_hl_test_configuration
    # WHEN using the Prompt decorator with default values
    _call_llm_with_defaults(messages=call_llm_messages)
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
