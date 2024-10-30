import os
import random
import string
import time
from unittest.mock import patch

import pytest
from humanloop.decorators.flow import flow
from humanloop.decorators.prompt import prompt
from humanloop.decorators.tool import tool
from humanloop.otel.constants import HL_FILE_OT_KEY, HL_TRACE_METADATA_KEY
from humanloop.otel.exporter import HumanloopSpanExporter
from humanloop.otel.helpers import read_from_opentelemetry_span
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from opentelemetry.sdk.trace import Tracer
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@tool()
def _random_string() -> str:
    """Return a random string."""
    # NOTE: This is very basic; scope is to check if it's
    # picked up and included in the Flow Trace
    return "".join(random.choices(string.ascii_letters + string.digits, k=10))


@prompt(path=None, template="You are an assistant on the following topics: {topics}.")
def _call_llm(messages: list[ChatCompletionMessageParam]) -> str:
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
    ) + _random_string()


def _agent_call_no_decorator(messages: list[dict]) -> str:
    return _call_llm(messages=messages)


@flow(attributes={"foo": "bar", "baz": 7})
def _agent_call(messages: list[dict]) -> str:
    return _call_llm(messages=messages)


@flow()
def _flow_over_flow(messages: list[dict]) -> str:
    return _agent_call(messages=messages)


def test_decorators_without_flow(
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    # GIVEN a call to @prompt annotated function that calls a @tool
    _, exporter = opentelemetry_hl_test_configuration
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
    spans = exporter.get_finished_spans()
    # THEN 2 independent spans are exported with no relation to each other
    assert len(spans) == 2
    assert read_from_opentelemetry_span(span=spans[0], key=HL_FILE_OT_KEY)["tool"]
    assert read_from_opentelemetry_span(span=spans[1], key=HL_FILE_OT_KEY)["prompt"]
    for span in spans:
        # THEN no metadata related to trace is present on either of them
        with pytest.raises(KeyError):
            read_from_opentelemetry_span(span=span, key=HL_TRACE_METADATA_KEY)


def test_decorators_with_flow_decorator(
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    # GIVEN a @flow entrypoint to an instrumented application
    _, exporter = opentelemetry_hl_test_configuration
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
    # THEN 3 spans are created
    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    # THEN the span are returned bottom to top
    assert read_from_opentelemetry_span(span=spans[0], key=HL_FILE_OT_KEY)["tool"]
    assert read_from_opentelemetry_span(span=spans[1], key=HL_FILE_OT_KEY)["prompt"]
    assert read_from_opentelemetry_span(span=spans[2], key=HL_FILE_OT_KEY)["flow"]
    tool_trace_metadata = read_from_opentelemetry_span(span=spans[0], key=HL_TRACE_METADATA_KEY)
    prompt_trace_metadata = read_from_opentelemetry_span(span=spans[1], key=HL_TRACE_METADATA_KEY)
    flow_trace_metadata = read_from_opentelemetry_span(span=spans[2], key=HL_TRACE_METADATA_KEY)
    # THEN Tool span is a child of Prompt span
    assert tool_trace_metadata["trace_parent_id"] == spans[1].context.span_id
    assert tool_trace_metadata["is_flow_log"] is False
    assert prompt_trace_metadata["trace_parent_id"] == spans[2].context.span_id
    # THEN Prompt span is a child of Flow span
    assert prompt_trace_metadata["is_flow_log"] is False
    assert flow_trace_metadata["is_flow_log"]
    assert flow_trace_metadata["trace_id"] == spans[2].context.span_id


def test_flow_decorator_flow_in_flow(
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
    call_llm_messages: list[dict],
):
    # GIVEN A configured OpenTelemetry tracer and exporter
    _, exporter = opentelemetry_hl_test_configuration

    # WHEN Calling the _test_flow_in_flow function with specific messages
    _flow_over_flow(call_llm_messages)

    # THEN Spans correctly produce a Flow Trace
    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    assert read_from_opentelemetry_span(span=spans[0], key=HL_FILE_OT_KEY)["tool"]
    assert read_from_opentelemetry_span(span=spans[1], key=HL_FILE_OT_KEY)["prompt"]
    assert read_from_opentelemetry_span(span=spans[2], key=HL_FILE_OT_KEY)["flow"]
    assert read_from_opentelemetry_span(span=spans[3], key=HL_FILE_OT_KEY)["flow"]

    tool_trace_metadata = read_from_opentelemetry_span(span=spans[0], key=HL_TRACE_METADATA_KEY)
    prompt_trace_metadata = read_from_opentelemetry_span(span=spans[1], key=HL_TRACE_METADATA_KEY)
    nested_flow_trace_metadata = read_from_opentelemetry_span(span=spans[2], key=HL_TRACE_METADATA_KEY)
    flow_trace_metadata = read_from_opentelemetry_span(span=spans[3], key=HL_TRACE_METADATA_KEY)
    # THEN the nested flow points to the parent flow
    assert tool_trace_metadata["trace_parent_id"] == spans[1].context.span_id
    assert tool_trace_metadata["is_flow_log"] is False
    assert prompt_trace_metadata["trace_parent_id"] == spans[2].context.span_id
    assert prompt_trace_metadata["is_flow_log"] is False
    assert nested_flow_trace_metadata["trace_id"] == spans[2].context.span_id
    # THEN the parent flow correctly points to itself
    assert nested_flow_trace_metadata["is_flow_log"]
    assert nested_flow_trace_metadata["trace_parent_id"] == spans[3].context.span_id
    assert flow_trace_metadata["is_flow_log"]
    assert flow_trace_metadata["trace_id"] == spans[3].context.span_id


def test_flow_decorator_with_hl_exporter(
    call_llm_messages: list[dict],
    opentelemetry_hl_with_exporter_test_configuration: tuple[Tracer, HumanloopSpanExporter],
):
    # NOTE: type ignore comments are caused by the MagicMock used to mock _client
    # GIVEN a OpenTelemetry configuration with a mock Humanloop SDK and a spied exporter
    _, exporter = opentelemetry_hl_with_exporter_test_configuration
    with patch.object(exporter, "export", wraps=exporter.export) as mock_export_method:
        # WHEN calling the @flow decorated function
        _agent_call(call_llm_messages)
        assert len(mock_export_method.call_args_list) == 3
        first_exported_span = mock_export_method.call_args_list[0][0][0][0]
        middle_exported_span = mock_export_method.call_args_list[1][0][0][0]
        last_exported_span = mock_export_method.call_args_list[2][0][0][0]
        # THEN the last uploaded span is the Flow
        assert read_from_opentelemetry_span(span=last_exported_span, key=HL_FILE_OT_KEY)["flow"]["attributes"] == {  # type: ignore[index,call-overload]
            "foo": "bar",
            "baz": 7,
        }
        # THEN the second uploaded span is the Prompt
        assert "prompt" in read_from_opentelemetry_span(span=middle_exported_span, key=HL_FILE_OT_KEY)
        # THEN the first uploaded span is the Tool
        assert "tool" in read_from_opentelemetry_span(span=first_exported_span, key=HL_FILE_OT_KEY)

        # Potentially flaky: Exporter is threaded, need
        # to wait for them to finish
        time.sleep(3)

        # THEN the first Log uploaded is the Flow
        first_log = exporter._client.flows.log.call_args_list[0][1]  # type: ignore[attr-defined]
        assert "flow" in first_log
        exporter._client.flows.log.assert_called_once()  # type: ignore[attr-defined]
        flow_log_call_args = exporter._client.flows.log.call_args_list[0]  # type: ignore[attr-defined]
        flow_log_call_args.kwargs["flow"]["attributes"] == {"foo": "bar", "baz": 7}
        flow_log_id = exporter._client.flows.log.return_value  # type: ignore[attr-defined]

        # THEN the second Log uploaded is the Prompt
        exporter._client.prompts.log.assert_called_once()  # type: ignore[attr-defined]
        prompt_log_call_args = exporter._client.prompts.log.call_args_list[0]  # type: ignore[attr-defined]
        prompt_log_call_args.kwargs["trace_parent_id"] == flow_log_id
        prompt_log_call_args.kwargs["prompt"]["temperature"] == 0.8
        prompt_log_id = exporter._client.prompts.log.return_value  # type: ignore[attr-defined]

        # THEN the final Log uploaded is the Tool
        exporter._client.tools.log.assert_called_once()  # type: ignore[attr-defined]
        tool_log_call_args = exporter._client.tools.log.call_args_list[0]  # type: ignore[attr-defined]
        tool_log_call_args.kwargs["trace_parent_id"] == prompt_log_id


def test_flow_decorator_hl_exporter_flow_inside_flow(
    call_llm_messages: list[dict],
    opentelemetry_hl_with_exporter_test_configuration: tuple[Tracer, HumanloopSpanExporter],
):
    # GIVEN a OpenTelemetry configuration with a mock Humanloop SDK and a spied exporter
    _, exporter = opentelemetry_hl_with_exporter_test_configuration
    with patch.object(exporter, "export", wraps=exporter.export) as mock_export_method:
        # WHEN calling the @flow decorated function
        _flow_over_flow(call_llm_messages)
        assert len(mock_export_method.call_args_list) == 4
        # THEN the last uploaded span is the larger Flow
        # THEN the second to last uploaded span is the nested Flow
        last_exported_span = mock_export_method.call_args_list[3][0][0][0]
        previous_exported_span = mock_export_method.call_args_list[2][0][0][0]
        last_span_flow_metadata = read_from_opentelemetry_span(span=last_exported_span, key=HL_TRACE_METADATA_KEY)
        previous_span_flow_metadata = read_from_opentelemetry_span(
            span=previous_exported_span, key=HL_TRACE_METADATA_KEY
        )
        assert previous_span_flow_metadata["trace_parent_id"] == last_exported_span.context.span_id
        assert last_span_flow_metadata["is_flow_log"]
        assert previous_span_flow_metadata["is_flow_log"]
