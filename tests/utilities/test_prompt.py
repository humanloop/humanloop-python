import os
import time
from typing import Any, Callable, Optional

import cohere
import pytest

# replicate has no typing stubs
import replicate  # type: ignore
from anthropic import Anthropic
from anthropic.types.message_param import MessageParam
from dotenv import load_dotenv
from groq import Groq
from groq import NotFoundError as GroqNotFoundError
from humanloop.client import Humanloop
from humanloop.eval_utils.run import HumanloopUtilityError
from humanloop.utilities.prompt import prompt
from humanloop.otel.constants import HUMANLOOP_FILE_KEY
from humanloop.otel.helpers import is_humanloop_span, read_from_opentelemetry_span
from humanloop.types.model_providers import ModelProviders
from humanloop.types.prompt_kernel_request import PromptKernelRequest
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from opentelemetry.sdk.trace import Tracer
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from replicate.exceptions import ModelError as ReplicateModelError  # type: ignore [import]

from tests.conftest import DirectoryIdentifiers  # type: ignore [import]


pytest.skip("skip for demo", allow_module_level=True)

_PROVIDER_AND_MODEL = [
    ("openai", "gpt-4o-mini"),
    ("groq", "llama3-8b-8192"),
    ("cohere", "command"),
    ("replicate", "meta/meta-llama-3-8b-instruct"),
    ("anthropic", "claude-3-haiku-20240307"),
]


def _test_scenario(opentelemetry_tracer: Tracer, **kwargs):
    """
    Set up the function decorated with @prompt.

    Normally the opentelemetry_tracer would be passed in by the Humanloop client.
    In a test environment, the Tracer is obtained from a fixture and the test
    call this function to setup the decorated function that is tested.
    """

    @prompt(opentelemetry_tracer=opentelemetry_tracer, **kwargs)
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
                # NOTE: Tests in this file are integration tests that rely on live LLM provider
                # clients. If a test fails, it might be flaky. If this happens, consider adding
                # a skip mechanism similar to Groq
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
            # This should be documented somewhere or changed
            replicate.default_client._api_token = os.getenv("REPLICATE_API_KEY")
            try:
                output = ""
                for event in replicate.run(
                    model,
                    input={
                        "prompt": messages[0]["content"] + " " + messages[-1]["content"],
                        "temperature": 0.8,
                    },
                ):
                    output += str(event)
            except ReplicateModelError:
                pytest.skip("Replicate not available")
            if not output:
                pytest.skip("Replicate not available")
            return output
        raise ValueError(f"Unknown provider: {provider}")

    return _call_llm_base


# LLM provider might not be available, retry the test
@pytest.mark.parametrize("provider_model", _PROVIDER_AND_MODEL)
def test_prompt_decorator(
    provider_model: tuple[str, str],
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
    call_llm_messages: list[ChatCompletionMessageParam],
):
    provider, model = provider_model
    # GIVEN an OpenTelemetry configuration without HumanloopSpanProcessor
    tracer, exporter = opentelemetry_test_configuration
    # WHEN using the Prompt decorator

    call_llm = _test_scenario(tracer)

    call_llm(
        provider=provider,
        model=model,
        messages=call_llm_messages,
    )

    # Wait for the Prompt span to be exported, it is waiting
    # asynchronously for the LLM provider call span to finish
    time.sleep(10)

    # THEN two spans are created: one for the OpenAI LLM provider call and one for the Prompt
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert not is_humanloop_span(span=spans[0])
    assert is_humanloop_span(span=spans[1])
    # THEN the Prompt span is not enhanced with information from the LLM provider
    assert is_humanloop_span(spans[1])
    # THEN no information is added to the Prompt span without the HumanloopSpanProcessor
    assert spans[1].attributes.get("prompt") is None  # type: ignore


# LLM provider might not be available, retry the test
@pytest.mark.flaky(retries=3, delay=60)
@pytest.mark.parametrize("provider_model", _PROVIDER_AND_MODEL)
def test_prompt_decorator_with_hl_processor(
    provider_model: tuple[str, str],
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
    call_llm_messages: list[ChatCompletionMessageParam],
):
    provider, model = provider_model
    # GIVEN an OpenTelemetry configuration with HumanloopSpanProcessor
    tracer, exporter = opentelemetry_hl_test_configuration
    # WHEN using the Prompt decorator

    call_llm = _test_scenario(opentelemetry_tracer=tracer)

    call_llm(
        provider=provider,
        model=model,
        messages=call_llm_messages,
    )

    # THEN two spans are created: one for the OpenAI LLM provider call and one for the Prompt

    # Wait for the Prompt span to be exported, it is waiting
    # asynchronously for the LLM provider call span to finish
    time.sleep(10)

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert not is_humanloop_span(span=spans[0])
    assert is_humanloop_span(span=spans[1])
    # THEN the Prompt span is enhanced with information and forms a correct PromptKernel
    prompt_kernel = PromptKernelRequest.model_validate(
        read_from_opentelemetry_span(
            span=spans[1],
            key=HUMANLOOP_FILE_KEY,
        )["prompt"]  # type: ignore
    )
    # THEN temperature is intercepted from LLM provider call
    assert prompt_kernel.temperature == 0.8
    # THEN the provider intercepted from LLM provider call
    assert prompt_kernel.provider == provider
    # THEN model is intercepted from LLM provider call
    assert prompt_kernel.model == model
    # THEN top_p is not present since it's not present in the LLM provider call
    assert prompt_kernel.top_p is None


# LLM provider might not be available, retry the test
@pytest.mark.flaky(retries=3, delay=60)
@pytest.mark.parametrize("provider_model", _PROVIDER_AND_MODEL)
def test_prompt_decorator_with_defaults(
    provider_model: tuple[str, str],
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
    call_llm_messages: list[ChatCompletionMessageParam],
):
    provider, model = provider_model
    # GIVEN an OpenTelemetry configuration with HumanloopSpanProcessor
    tracer, exporter = opentelemetry_hl_test_configuration
    # WHEN using the Prompt decorator with default values

    call_llm = _test_scenario(
        opentelemetry_tracer=tracer,
        temperature=0.9,
        top_p=0.1,
        template="You are an assistant on the following topics: {topics}.",
        path=None,
    )

    call_llm(
        provider=provider,
        model=model,
        messages=call_llm_messages,
    )

    # Wait for the Prompt span to be exported, it is waiting
    # asynchronously for the LLM provider call span to finish
    time.sleep(10)

    spans = exporter.get_finished_spans()
    # THEN the Prompt span is enhanced with information and forms a correct PromptKernel
    prompt = PromptKernelRequest.model_validate(
        read_from_opentelemetry_span(span=spans[1], key=HUMANLOOP_FILE_KEY)["prompt"]  # type: ignore
    )
    # THEN temperature intercepted from LLM provider call is overridden by default value
    assert prompt.temperature == 0.9
    # THEN top_p is taken from decorator default value
    assert prompt.top_p == 0.1
    # THEN the provider intercepted from LLM provider call
    assert prompt.model == model


# LLM provider might not be available, retry the test
@pytest.mark.flaky(retries=3, delay=60)
@pytest.mark.parametrize(
    "attributes_test_expected",
    [
        (
            {"foo": "bar"},
            {"foo": "bar"},
        ),
        (
            {},
            None,
        ),
        (
            None,
            None,
        ),
    ],
)
def test_prompt_attributes(
    attributes_test_expected: tuple[dict[str, str], dict[str, str]],
    call_llm_messages: list[ChatCompletionMessageParam],
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    test_attributes, expected_attributes = attributes_test_expected
    tracer, exporter = opentelemetry_hl_test_configuration

    call_llm = _test_scenario(
        opentelemetry_tracer=tracer,
        path=None,
        attributes=test_attributes,
    )

    call_llm(
        provider="openai",
        model="gpt-4o",
        messages=call_llm_messages,
    )

    # Wait for the Prompt span to be exported, it is waiting
    # asynchronously for the LLM provider call span to finish
    time.sleep(10)

    assert len(exporter.get_finished_spans()) == 2

    prompt_kernel = PromptKernelRequest.model_validate(
        read_from_opentelemetry_span(
            span=exporter.get_finished_spans()[1],
            key=HUMANLOOP_FILE_KEY,
        )["prompt"]  # type: ignore
    )
    assert prompt_kernel.attributes == expected_attributes


def test_prompt_decorator_with_hl_call(
    humanloop_client: Humanloop,
    test_directory: DirectoryIdentifiers,
    get_test_path: Callable[[str], str],
):
    file_path = get_test_path("Test Prompt")

    @humanloop_client.prompt(path=file_path)
    def call_llm_with_hl_call(messages: list[dict]):
        response = humanloop_client.prompts.call(  # type: ignore [call-args]
            path=file_path,
            messages=messages,  # type: ignore [arg-type]
            prompt={
                "model": "gpt-4o-mini",
                "temperature": 0.8,
            },
        )
        return response.logs[0].output_message.content  # type: ignore [union-attr]

    output = call_llm_with_hl_call(
        messages=[
            {
                "role": "user",
                "content": "Hi!",
            },
        ]
    )

    assert output is not None
    response = humanloop_client.directories.get(id=test_directory.id)
    prompt = [file for file in response.files if file.path == file_path][0]
    assert prompt.path == file_path
    response = humanloop_client.logs.list(file_id=prompt.id)  # type: ignore [assignment]
    assert len(response.items) == 1  # type: ignore


@pytest.mark.skip("prompt.call() unhandled behavior")
def test_overridden_call_with_prompt_in_prompt(
    humanloop_client: Humanloop,
    test_directory: DirectoryIdentifiers,
    get_test_path: Callable[[str], str],
):
    inner_file_path = get_test_path("Test Prompt")
    outer_file_path = get_test_path("Outer Test Prompt")

    @humanloop_client.prompt(path=inner_file_path)
    def call_llm_with_hl_call(messages: list[dict]):
        response = humanloop_client.prompts.call(  # type: ignore [call-args]
            path=inner_file_path,
            messages=messages,  # type: ignore [arg-type]
            prompt={
                "model": "gpt-4o-mini",
                "temperature": 0.8,
            },
        )
        return response.logs[0].output_message.content  # type: ignore [union-attr]

    @humanloop_client.prompt(path=outer_file_path)
    def outer_call_llm_with_hl_call(messages: list[dict]):
        output = call_llm_with_hl_call(messages)
        response = humanloop_client.prompts.call(  # type: ignore [call-args]
            path=outer_file_path,
            messages=[
                {
                    "role": "user",
                    "content": f"Give a clever response to this {output}",
                }
            ],
            prompt={
                "model": "gpt-4o-mini",
                "temperature": 0.8,
            },
        )
        return response.logs[0].output_message.content  # type: ignore [union-attr]

    output = outer_call_llm_with_hl_call(
        messages=[
            {
                "role": "user",
                "content": "Hi!",
            },
        ]
    )

    # Wait for the workspace to be updated
    time.sleep(10)

    assert output is not None
    response = humanloop_client.directories.get(id=test_directory.id)
    outer_prompt = [file for file in response.files if file.path == outer_file_path][0]
    inner_prompt = [file for file in response.files if file.path == inner_file_path][0]

    assert outer_prompt.path == outer_file_path
    response = humanloop_client.logs.list(file_id=outer_prompt.id)  # type: ignore [assignment]
    assert len(response.items) == 1  # type: ignore [attr-defined]

    assert inner_prompt.path == inner_file_path
    response = humanloop_client.logs.list(file_id=inner_prompt.id)  # type: ignore [assignment]
    assert len(response.items) == 1  # type: ignore [attr-defined]


def test_overridden_call_fails_obviously(
    humanloop_client: Humanloop,
    test_directory: DirectoryIdentifiers,
    get_test_path: Callable[[str], str],
):
    file_path = get_test_path("Test Prompt")

    @humanloop_client.prompt(path=file_path)
    def call_llm_with_hl_call():
        response = humanloop_client.prompts.call(  # type: ignore [call-args]
            path=file_path,
            messages={
                "role": "system",
                "content": "This fails because messages should be a list",
            },  # type: ignore [arg-type]
            prompt={
                "model": "gpt-4o-mini",
                "temperature": 0.8,
            },
        )
        return response.logs[0].output_message.content  # type: ignore [union-attr]

    with pytest.raises(HumanloopUtilityError):
        call_llm_with_hl_call()

    response = humanloop_client.directories.get(id=test_directory.id)
    assert not any(file.path == file_path for file in response.files)


def test_overridden_call_must_match_utility_path(
    humanloop_client: Humanloop,
    test_directory: DirectoryIdentifiers,
    get_test_path: Callable[[str], str],
):
    @humanloop_client.prompt(
        path=get_test_path("Test Prompt"),
        temperature=0.7,
    )
    def call_llm_with_hl_call():
        response = humanloop_client.prompts.call(
            path=get_test_path("Test Prompt 2"),
            prompt={
                "model": "gpt-4o-mini",
            },
            messages=[
                {
                    "role": "user",
                    "content": "How are you?",
                }
            ],
        )

        return response.logs[0].output_message.content

    with pytest.raises(HumanloopUtilityError):
        call_llm_with_hl_call()

    response = humanloop_client.directories.get(id=test_directory.id)
    assert not any(file.path == get_test_path("Test Prompt") for file in response.files)


@pytest.mark.parametrize(
    "key,utility_value,call_value",
    [
        # TODO: Bug found in backend: not specifying a model 400s but creates a File
        # ("provider", "openai", "anthropic"),
        ("temperature", 0.8, 0.5),
        ("top_p", 0.5, 0.3),
        ("stop", "foo", "bar"),
        ("presence_penalty", 0.7, 0.5),
        ("frequency_penalty", 1.5, 1),
        ("other", "foo", "bar"),
        ("seed", 42, 43),
        # TODO: Bug found in backend: not specifying a model 400s but creates a File
        # ("response_format", {"type": "json_object"}, {"type": "json_schema"}),
    ],
)
def test_overridden_call_must_match_utility(
    key: str,
    utility_value: Any,
    call_value: Any,
    humanloop_client: Humanloop,
    get_test_path: Callable[[str], str],
    test_directory: DirectoryIdentifiers,
):
    path = get_test_path("Test Prompt")

    @humanloop_client.prompt(path=path, **{key: utility_value})
    def call_llm_with_hl_call():
        response = humanloop_client.prompts.call(
            path=path,
            prompt={
                "model": "gpt-4o-mini",
                **{key: call_value},
            },
            messages=[
                {
                    "role": "user",
                    "content": "How are you?",
                }
            ],
        )

        return response.logs[0].output_message.content

    with pytest.raises(HumanloopUtilityError):
        call_llm_with_hl_call()

    response = humanloop_client.directories.get(id=test_directory.id)
    assert not any(file.path == path for file in response.files)


@pytest.mark.parametrize(
    "key,prompt_call_value",
    [
        ("temperature", 0.5),
        ("top_p", 0.3),
        ("stop", "bar"),
        ("presence_penalty", 0.5),
        ("frequency_penalty", 1),
        ("seed", 42),
    ],
)
def test_values_specified_cal_override_utility(
    key: str,
    prompt_call_value: Any,
    humanloop_client: Humanloop,
    get_test_path: Callable[[str], str],
    test_directory: DirectoryIdentifiers,
):
    path = get_test_path("Test Prompt")

    @humanloop_client.prompt(path=path)
    def call_llm_with_hl_call():
        response = humanloop_client.prompts.call(
            path=path,
            prompt={
                "model": "gpt-4o-mini",
                **{key: prompt_call_value},
            },
            messages=[
                {
                    "role": "user",
                    "content": "How are you?",
                }
            ],
        )

        return response.logs[0].output_message.content

    call_llm_with_hl_call()

    response = humanloop_client.directories.get(id=test_directory.id)
    prompt = [file for file in response.files if file.path == path][0]
    assert getattr(prompt, key) == prompt_call_value
