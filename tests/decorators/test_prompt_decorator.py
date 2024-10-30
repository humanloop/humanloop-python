import os
from typing import Optional

import cohere
import pytest

# replicate has no typing stubs
import replicate  # type: ignore
from anthropic import Anthropic
from anthropic.types.message_param import MessageParam
from dotenv import load_dotenv
from groq import Groq
from groq import NotFoundError as GroqNotFoundError
from humanloop.decorators.prompt import prompt
from humanloop.otel.constants import HL_FILE_OT_KEY
from humanloop.otel.helpers import is_humanloop_span, read_from_opentelemetry_span
from humanloop.types.model_providers import ModelProviders
from humanloop.types.prompt_kernel_request import PromptKernelRequest
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from opentelemetry.sdk.trace import Tracer
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_PROVIDER_AND_MODEL = [
    ("openai", "gpt-4o"),
    ("groq", "llama3-8b-8192"),
    ("cohere", "command"),
    ("replicate", "meta/meta-llama-3-8b-instruct"),
    ("anthropic", "claude-3-opus-latest"),
]


def _call_llm_base(provider: ModelProviders, model: str, messages: list[dict]) -> Optional[str]:
    load_dotenv()
    if provider == "openai":
        # NOTE: These tests check if instrumentors are capable of intercepting OpenAI
        # provider calls. Could not find a way to intercept them coming from a Mock.
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # type: ignore
        return (
            client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                temperature=0.8,
            )
            .choices[0]
            .message.content
        )
    if provider == "anthropic":
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))  # type: ignore
        messages_anthropic_format = [
            MessageParam(
                content=message["content"],
                role="user" if message["role"] in ("user", "system") else "assistant",
            )
            for message in messages
        ]
        return (
            client.messages.create(  # type: ignore
                model=model,
                messages=messages_anthropic_format,
                max_tokens=200,
                temperature=0.8,
            )
            .content[0]
            .text
        )
    if provider == "groq":
        try:
            client = Groq(  # type: ignore
                # This is the default and can be omitted
                api_key=os.environ.get("GROQ_API_KEY"),
            )
            return (
                client.chat.completions.create(
                    messages=messages,  # type: ignore
                    model=model,
                    temperature=0.8,
                )
                .choices[0]
                .message.content
            )
        except GroqNotFoundError:
            pytest.skip("GROQ not available")
    if provider == "cohere":
        client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))  # type: ignore
        messages_cohere_format: list[cohere.Message] = []
        for message in messages:
            if message["role"] == "system":
                messages_cohere_format.append(cohere.SystemMessage(message=message["content"]))
            elif message["role"] == "user":
                messages_cohere_format.append(cohere.UserMessage(message=message["content"]))
            elif message["role"] == "assistant":
                messages_cohere_format.append(cohere.ChatbotMessage(message=message["content"]))
        return client.chat(  # type: ignore
            chat_history=messages_cohere_format,
            model=model,
            max_tokens=200,
            message=messages[-1]["content"],
            temperature=0.8,
        ).text
    if provider == "replicate":
        # TODO: Instrumentor only picks up methods on module-level, not client level
        replicate.default_client._api_token = os.getenv("REPLICATE_API_KEY")
        output = ""
        for event in replicate.run(
            model,
            input={
                "prompt": messages[0]["content"] + " " + messages[-1]["content"],
                "temperature": 0.8,
            },
        ):
            output += str(event)
        return output
    raise ValueError(f"Unknown provider: {provider}")


# NOTE: prompt is a decorator, but for brevity, it's used as a higher-order function in tests
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


@pytest.mark.parametrize("provider_model", _PROVIDER_AND_MODEL)
def test_prompt_decorator(
    provider_model: tuple[str, str],
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
    call_llm_messages: list[ChatCompletionMessageParam],
):
    provider, model = provider_model
    # GIVEN an OpenTelemetry configuration without HumanloopSpanProcessor
    _, exporter = opentelemetry_test_configuration
    # WHEN using the Prompt decorator
    _call_llm(
        provider=provider,
        model=model,
        messages=call_llm_messages,
    )
    # THEN two spans are created: one for the OpenAI LLM provider call and one for the Prompt
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert not is_humanloop_span(span=spans[0])
    assert is_humanloop_span(span=spans[1])
    # THEN the Prompt span is not enhanced with information from the LLM provider
    assert is_humanloop_span(spans[1])
    # THEN no information is added to the Prompt span without the HumanloopSpanProcessor
    assert spans[1].attributes.get("prompt") is None  # type: ignore


@pytest.mark.parametrize("provider_model", _PROVIDER_AND_MODEL)
def test_prompt_decorator_with_hl_processor(
    provider_model: tuple[str, str],
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
    call_llm_messages: list[ChatCompletionMessageParam],
):
    provider, model = provider_model
    # GIVEN an OpenTelemetry configuration with HumanloopSpanProcessor
    _, exporter = opentelemetry_hl_test_configuration
    # WHEN using the Prompt decorator
    _call_llm(
        provider=provider,
        model=model,
        messages=call_llm_messages,
    )
    # THEN two spans are created: one for the OpenAI LLM provider call and one for the Prompt
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert not is_humanloop_span(span=spans[0])
    assert is_humanloop_span(span=spans[1])
    # THEN the Prompt span is enhanced with information and forms a correct PromptKernel
    prompt = PromptKernelRequest.model_validate(
        read_from_opentelemetry_span(
            span=spans[1],
            key=HL_FILE_OT_KEY,
        )["prompt"]  # type: ignore
    )
    # THEN temperature is intercepted from LLM provider call
    assert prompt.temperature == 0.8
    # THEN the provider intercepted from LLM provider call
    assert prompt.provider == provider
    # THEN model is intercepted from LLM provider call
    assert prompt.model == model
    # THEN top_p is not present since it's not present in the LLM provider call
    assert prompt.top_p is None


@pytest.mark.parametrize("provider_model", _PROVIDER_AND_MODEL)
def test_prompt_decorator_with_defaults(
    provider_model: tuple[str, str],
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
    call_llm_messages: list[ChatCompletionMessageParam],
):
    provider, model = provider_model
    # GIVEN an OpenTelemetry configuration with HumanloopSpanProcessor
    _, exporter = opentelemetry_hl_test_configuration
    # WHEN using the Prompt decorator with default values
    _call_llm_with_defaults(
        provider=provider,
        model=model,
        messages=call_llm_messages,
    )
    spans = exporter.get_finished_spans()
    # THEN the Prompt span is enhanced with information and forms a correct PromptKernel
    prompt = PromptKernelRequest.model_validate(
        read_from_opentelemetry_span(span=spans[1], key=HL_FILE_OT_KEY)["prompt"]  # type: ignore
    )
    # THEN temperature intercepted from LLM provider call is overridden by default value
    assert prompt.temperature == 0.9
    # THEN top_p is taken from decorator default value
    assert prompt.top_p == 0.1
    # THEN the provider intercepted from LLM provider call
    assert prompt.model == model


@pytest.mark.parametrize(
    "hyperparameters",
    (
        {"temperature": 1.1},
        {"top_p": 1.1},
        {"presence_penalty": 3},
        {"frequency_penalty": 3},
    ),
)
def test_hyperparameter_values_fail_out_of_domain(hyperparameters: dict[str, float]):
    # GIVEN a Prompt decorated function

    with pytest.raises(ValueError):
        # WHEN using default values that are out of domain
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

    # THEN an exception is raised
