import os
import random
import string
import time
from unittest.mock import patch

from humanloop.decorators.flow import flow
from humanloop.decorators.prompt import prompt
from humanloop.decorators.tool import tool
from humanloop.otel import TRACE_FLOW_CONTEXT
from humanloop.otel.constants import HL_FILE_OT_KEY
from humanloop.otel.exporter import HumanloopSpanExporter
from humanloop.otel.helpers import read_from_opentelemetry_span
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from opentelemetry.sdk.trace import Tracer
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


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

    @prompt(
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

    @flow(
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
    spans = exporter.get_finished_spans()
    # THEN 3 spans arrive at the exporter in the following order:
    #   0. Intercepted OpenAI call, which is ignored by the exporter
    #   1. Tool Span (called after the OpenAI call but before the Prompt Span finishes)
    #   2. Prompt Span
    assert len(spans) == 3
    assert read_from_opentelemetry_span(
        span=spans[1],
        key=HL_FILE_OT_KEY,
    )["tool"]
    assert read_from_opentelemetry_span(
        span=spans[2],
        key=HL_FILE_OT_KEY,
    )["prompt"]
    for span in spans:
        # THEN no metadata related to trace is present on either of them
        assert TRACE_FLOW_CONTEXT.get(span.get_span_context().span_id) is None


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
    # THEN 4 spans arrive at the exporter in the following order:
    #   0. Intercepted OpenAI call, which is ignored by the exporter
    #   1. Tool Span (called after the OpenAI call but before the Prompt Span finishes)
    #   2. Prompt Span
    #   3. Flow Span
    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    # THEN the span are returned bottom to top
    assert read_from_opentelemetry_span(span=spans[1], key=HL_FILE_OT_KEY)["tool"]
    assert read_from_opentelemetry_span(span=spans[2], key=HL_FILE_OT_KEY)["prompt"]
    assert read_from_opentelemetry_span(span=spans[3], key=HL_FILE_OT_KEY)["flow"]
    tool_trace_metadata = TRACE_FLOW_CONTEXT.get(spans[1].get_span_context().span_id)
    prompt_trace_metadata = TRACE_FLOW_CONTEXT.get(spans[2].get_span_context().span_id)
    flow_trace_metadata = TRACE_FLOW_CONTEXT.get(spans[3].get_span_context().span_id)
    # THEN Tool span is a child of Prompt span
    assert tool_trace_metadata["trace_parent_id"] == spans[2].context.span_id
    assert tool_trace_metadata["is_flow_log"] is False
    assert prompt_trace_metadata["trace_parent_id"] == spans[3].context.span_id
    # THEN Prompt span is a child of Flow span
    assert prompt_trace_metadata["is_flow_log"] is False
    assert flow_trace_metadata["is_flow_log"]
    assert flow_trace_metadata["trace_id"] == spans[3].context.span_id


def test_flow_decorator_flow_in_flow(
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
    call_llm_messages: list[dict],
):
    # GIVEN A configured OpenTelemetry tracer and exporter
    tracer, exporter = opentelemetry_hl_test_configuration

    _flow_over_flow = _test_scenario(tracer)[3]

    # WHEN Calling the _test_flow_in_flow function with specific messages
    _flow_over_flow(call_llm_messages)

    # THEN 5 spans are arrive at the exporter in the following order:
    #   0. Intercepted OpenAI call, which is ignored by the exporter
    #   1. Tool Span (called after the OpenAI call but before the Prompt Span finishes)
    #   2. Prompt Span
    #   3. Nested Flow Span
    #   4. Flow Span
    spans = exporter.get_finished_spans()
    assert len(spans) == 5
    assert read_from_opentelemetry_span(span=spans[1], key=HL_FILE_OT_KEY)["tool"]
    assert read_from_opentelemetry_span(span=spans[2], key=HL_FILE_OT_KEY)["prompt"]
    assert read_from_opentelemetry_span(span=spans[3], key=HL_FILE_OT_KEY)["flow"]
    assert read_from_opentelemetry_span(span=spans[4], key=HL_FILE_OT_KEY)["flow"]

    tool_trace_metadata = TRACE_FLOW_CONTEXT.get(spans[1].get_span_context().span_id)
    prompt_trace_metadata = TRACE_FLOW_CONTEXT.get(spans[2].get_span_context().span_id)
    nested_flow_trace_metadata = TRACE_FLOW_CONTEXT.get(spans[3].get_span_context().span_id)
    flow_trace_metadata = TRACE_FLOW_CONTEXT.get(spans[4].get_span_context().span_id)
    # THEN the parent of the Tool Log is the Prompt Log
    assert tool_trace_metadata["trace_parent_id"] == spans[2].context.span_id
    assert tool_trace_metadata["is_flow_log"] is False
    # THEN the parent of the Prompt Log is the Flow Log
    assert prompt_trace_metadata["trace_parent_id"] == spans[3].context.span_id
    assert prompt_trace_metadata["is_flow_log"] is False
    # THEN the nested Flow Log creates a new trace
    assert nested_flow_trace_metadata["trace_id"] == spans[3].context.span_id
    assert nested_flow_trace_metadata["is_flow_log"]
    # THEN the parent of the nested Flow Log is the upper Flow Log
    assert nested_flow_trace_metadata["trace_parent_id"] == spans[4].context.span_id
    # THEN the parent Flow Log correctly points to itself
    assert flow_trace_metadata["trace_id"] == spans[4].context.span_id
    assert flow_trace_metadata["is_flow_log"]


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

        # THEN 4 spans are arrive at the exporter in the following order:
        #   0. Intercepted OpenAI call, which is ignored by the exporter
        #   1. Tool Span (called after the OpenAI call but before the Prompt Span finishes)
        #   2. Prompt Span
        #   3. Flow Span
        assert len(mock_export_method.call_args_list) == 4

        tool_span = mock_export_method.call_args_list[1][0][0][0]
        prompt_span = mock_export_method.call_args_list[2][0][0][0]
        flow_span = mock_export_method.call_args_list[3][0][0][0]
        # THEN the last uploaded span is the Flow
        assert read_from_opentelemetry_span(
            span=flow_span,
            key=HL_FILE_OT_KEY,
        )["flow"]["attributes"] == {  # type: ignore[index,call-overload]
            "foo": "bar",
            "baz": 7,
        }
        # THEN the second uploaded span is the Prompt
        assert "prompt" in read_from_opentelemetry_span(
            span=prompt_span,
            key=HL_FILE_OT_KEY,
        )
        # THEN the first uploaded span is the Tool
        assert "tool" in read_from_opentelemetry_span(
            span=tool_span,
            key=HL_FILE_OT_KEY,
        )

        # NOTE: The type: ignore comments are caused by the MagicMock used to mock the HTTP client

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

        # THEN 5 spans are arrive at the exporter in the following order:
        #   0. Intercepted OpenAI call, which is ignored by the exporter
        #   1. Tool Span (called after the OpenAI call but before the Prompt Span finishes)
        #   2. Prompt Span
        #   3. Nested Flow Span
        #   4. Flow Span
        assert len(mock_export_method.call_args_list) == 5
        # THEN the last uploaded span is the larger Flow
        # THEN the second to last uploaded span is the nested Flow
        flow_span = mock_export_method.call_args_list[4][0][0][0]
        nested_flow_span = mock_export_method.call_args_list[3][0][0][0]
        last_span_flow_metadata = TRACE_FLOW_CONTEXT.get(flow_span.get_span_context().span_id)
        flow_span_flow_metadata = TRACE_FLOW_CONTEXT.get(nested_flow_span.get_span_context().span_id)
        assert flow_span_flow_metadata["trace_parent_id"] == flow_span.context.span_id
        assert last_span_flow_metadata["is_flow_log"]
        assert flow_span_flow_metadata["is_flow_log"]
