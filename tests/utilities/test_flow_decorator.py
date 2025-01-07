import os
import random
import string
import time

from unittest.mock import patch
import pytest
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from opentelemetry.sdk.trace import Tracer
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace import ReadableSpan

from humanloop.utilities.flow import flow
from humanloop.utilities.prompt import prompt
from humanloop.utilities.tool import tool
from humanloop.otel.constants import HUMANLOOP_FILE_KEY
from humanloop.otel.exporter import HumanloopSpanExporter
from humanloop.otel.helpers import read_from_opentelemetry_span


def _test_scenario(
    opentelemetry_tracer: Tracer,
):
    @tool(opentelemetry_tracer=opentelemetry_tracer)
    def _random_string() -> str:
        """Return a random string."""
        return "".join(
            random.choices(
                string.ascii_letters + string.digits,
                k=10,
            )
        )

    @prompt(  # type: ignore
        opentelemetry_tracer=opentelemetry_tracer,
        path=None,
        template="You are an assistant on the following topics: {topics}.",
    )
    def _call_llm(messages: list[ChatCompletionMessageParam]) -> str:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return (
            client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.8,
            )
            .choices[0]
            .message.content
        ) + _random_string()

    @flow(
        opentelemetry_tracer=opentelemetry_tracer,
        attributes={"foo": "bar", "baz": 7},
    )
    def _agent_call(messages: list[dict]) -> str:
        return _call_llm(messages=messages)

    @flow(  # type: ignore
        opentelemetry_tracer=opentelemetry_tracer,
    )
    def _flow_over_flow(messages: list[dict]) -> str:
        return _agent_call(messages=messages)

    return _random_string, _call_llm, _agent_call, _flow_over_flow


def test_decorators_without_flow(
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    tracer, exporter = opentelemetry_hl_test_configuration

    _call_llm = _test_scenario(tracer)[1]

    # GIVEN a call to @prompt annotated function that calls a @tool
    _call_llm(
        [
            {
                "role": "system",
                "content": "You are an assistant on the following topics: greetings in foreign languages.",
            },
            {
                "role": "user",
                "content": "Hello, how are you?",
            },
        ]
    )
    # WHEN exporting the spans
    # Wait for the prompt span to be exported; It was waiting
    # on the OpenAI call span to finish first
    time.sleep(1)
    spans = exporter.get_finished_spans()

    # THEN 3 spans arrive at the exporter:
    assert len(spans) == 3

    for i in range(3):
        if spans[i].name == "humanloop.tool":
            tool_span = spans[i]
        elif spans[i].name == "humanloop.prompt":
            prompt_span = spans[i]

    assert read_from_opentelemetry_span(
        span=tool_span,
        key=HUMANLOOP_FILE_KEY,
    )["tool"]
    assert read_from_opentelemetry_span(
        span=prompt_span,
        key=HUMANLOOP_FILE_KEY,
    )["prompt"]


def test_decorators_with_flow_decorator(
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    # GIVEN a @flow entrypoint to an instrumented application
    tracer, exporter = opentelemetry_hl_test_configuration

    _agent_call = _test_scenario(tracer)[2]

    # WHEN calling the Flow
    _agent_call(
        [
            {
                "role": "system",
                "content": "You are an assistant on the following topics: greetings in foreign languages.",
            },
            {
                "role": "user",
                "content": "Hello, how are you?",
            },
        ]
    )

    # THEN 4 spans arrive at the exporter:
    spans: list[ReadableSpan] = exporter.get_finished_spans()
    assert len(spans) == 4

    for i in range(4):
        if spans[i].name == "humanloop.flow":
            flow_span = spans[i]
        elif spans[i].name == "humanloop.prompt":
            prompt_span = spans[i]
        elif spans[i].name == "humanloop.tool":
            tool_span = spans[i]

    # THEN the span are returned bottom to top
    assert read_from_opentelemetry_span(span=tool_span, key=HUMANLOOP_FILE_KEY)["tool"]
    assert read_from_opentelemetry_span(span=prompt_span, key=HUMANLOOP_FILE_KEY)["prompt"]
    assert read_from_opentelemetry_span(span=flow_span, key=HUMANLOOP_FILE_KEY)["flow"]


def test_flow_decorator_flow_in_flow(
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
    call_llm_messages: list[dict],
):
    # GIVEN A configured OpenTelemetry tracer and exporter
    tracer, exporter = opentelemetry_hl_test_configuration

    _flow_over_flow = _test_scenario(tracer)[3]

    # WHEN Calling the _test_flow_in_flow function with specific messages
    _flow_over_flow(call_llm_messages)

    # Wait for the Prompt span to be exported; It was asynchronously waiting
    # on the OpenAI call span to finish first
    time.sleep(1)

    # THEN 5 spans arrive at the exporter
    spans: list[ReadableSpan] = exporter.get_finished_spans()
    assert len(spans) == 5

    for i in range(5):
        if spans[i].name == "humanloop.flow" and spans[i].parent is None:
            flow_span = spans[i]
        elif spans[i].name == "humanloop.flow" and spans[i].parent:
            nested_flow_span = spans[i]
        elif spans[i].name == "humanloop.prompt":
            prompt_span = spans[i]
        elif spans[i].name == "humanloop.tool":
            tool_span = spans[i]

    assert read_from_opentelemetry_span(span=tool_span, key=HUMANLOOP_FILE_KEY)["tool"]
    assert read_from_opentelemetry_span(span=prompt_span, key=HUMANLOOP_FILE_KEY)["prompt"]
    assert read_from_opentelemetry_span(span=nested_flow_span, key=HUMANLOOP_FILE_KEY)["flow"] != {}
    with pytest.raises(KeyError):
        read_from_opentelemetry_span(span=flow_span, key=HUMANLOOP_FILE_KEY)["flow"] != {}


def test_flow_decorator_with_hl_exporter(
    call_llm_messages: list[dict],
    opentelemetry_hl_with_exporter_test_configuration: tuple[Tracer, HumanloopSpanExporter],
):
    # NOTE: type ignore comments are caused by the MagicMock used to mock _client
    # GIVEN a OpenTelemetry configuration with a mock Humanloop SDK and a spied exporter
    tracer, exporter = opentelemetry_hl_with_exporter_test_configuration

    _agent_call = _test_scenario(tracer)[2]

    with patch.object(exporter, "export", wraps=exporter.export) as mock_export_method:
        # WHEN calling the @flow decorated function
        _agent_call(call_llm_messages)

        # Exporter is threaded, need to wait threads shutdown
        time.sleep(3)

        assert len(mock_export_method.call_args_list) == 4

        for i in range(4):
            span = mock_export_method.call_args_list[i][0][0][0]
            if span.name == "humanloop.flow":
                flow_span = span
            elif span.name == "humanloop.prompt":
                prompt_span = span
            elif span.name == "humanloop.tool":
                tool_span = span

        assert read_from_opentelemetry_span(
            span=flow_span,
            key=HUMANLOOP_FILE_KEY,
        )["flow"]["attributes"] == {  # type: ignore[index,call-overload]
            "foo": "bar",
            "baz": 7,
        }
        # THEN the second uploaded span is the Prompt
        assert "prompt" in read_from_opentelemetry_span(
            span=prompt_span,
            key=HUMANLOOP_FILE_KEY,
        )
        # THEN the first uploaded span is the Tool
        assert "tool" in read_from_opentelemetry_span(
            span=tool_span,
            key=HUMANLOOP_FILE_KEY,
        )

        # THEN the first Log uploaded is the Flow
        first_log = exporter._client.flows.log.call_args_list[0][1]  # type: ignore
        assert "flow" in first_log
        exporter._client.flows.log.assert_called_once()  # type: ignore
        flow_log_call_args = exporter._client.flows.log.call_args_list[0]  # type: ignore
        assert flow_log_call_args.kwargs["flow"]["attributes"] == {"foo": "bar", "baz": 7}
        flow_log_id = exporter._client.flows.log.return_value.id  # type: ignore

        # THEN the second Log uploaded is the Prompt
        exporter._client.prompts.log.assert_called_once()  # type: ignore
        prompt_log_call_args = exporter._client.prompts.log.call_args_list[0]  # type: ignore
        assert prompt_log_call_args.kwargs["trace_parent_id"] == flow_log_id
        assert prompt_log_call_args.kwargs["prompt"]["temperature"] == 0.8
        prompt_log_id = exporter._client.prompts.log.return_value.id  # type: ignore

        # THEN the final Log uploaded is the Tool
        exporter._client.tools.log.assert_called_once()  # type: ignore
        tool_log_call_args = exporter._client.tools.log.call_args_list[0]  # type: ignore
        assert tool_log_call_args.kwargs["trace_parent_id"] == prompt_log_id


def test_flow_decorator_hl_exporter_flow_inside_flow(
    call_llm_messages: list[dict],
    opentelemetry_hl_with_exporter_test_configuration: tuple[Tracer, HumanloopSpanExporter],
):
    # GIVEN a OpenTelemetry configuration with a mock Humanloop SDK and a spied exporter
    tracer, exporter = opentelemetry_hl_with_exporter_test_configuration

    _flow_over_flow = _test_scenario(tracer)[3]

    with patch.object(exporter, "export", wraps=exporter.export) as mock_export_method:
        # WHEN calling the @flow decorated function
        _flow_over_flow(call_llm_messages)

        # Exporter is threaded, need to wait threads shutdown
        time.sleep(3)

        # THEN 5 spans are arrive at the exporter
        assert len(mock_export_method.call_args_list) == 5

        # THEN one of the flows is nested inside the other
        spans: list[ReadableSpan] = [mock_export_method.call_args_list[i][0][0][0] for i in range(1, 5)]
        counter = 0
        for span in spans:
            if span.name == "humanloop.flow":
                counter += 1
                if span.parent:
                    nested_flow_span = span
                else:
                    flow_span = span
        # We are certain span_id exists for these 2 spans
        assert nested_flow_span.parent.span_id == flow_span.context.span_id  # type: ignore
