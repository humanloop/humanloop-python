from dataclasses import dataclass
import logging
from collections import defaultdict
from typing import Optional
import typing

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

from humanloop.base_client import BaseHumanloop
from humanloop.otel.constants import (
    HUMANLOOP_FILE_TYPE_KEY,
    HUMANLOOP_FLOW_PREREQUISITES_KEY,
    HUMANLOOP_INTERCEPTED_HL_CALL_SPAN_NAME,
    HUMANLOOP_LOG_KEY,
)
from humanloop.otel.helpers import (
    is_humanloop_span,
    is_llm_provider_call,
    write_to_opentelemetry_span,
)
from humanloop.otel.processor.prompts import enhance_prompt_span

if typing.TYPE_CHECKING:
    from humanloop.base_client import BaseHumanloop


logger = logging.getLogger("humanloop.sdk")


# NOTE: Source of bugs, refactor to dataclass for type safety
# Instead of accessing via "key"
@dataclass
class DependantSpan:
    span: ReadableSpan
    finished: bool


class HumanloopSpanProcessor(SimpleSpanProcessor):
    """Enrich Humanloop spans with data from their children spans.

    The decorators add Instrumentors to the OpenTelemetry TracerProvider
    that log interactions with common LLM libraries. These Instrumentors
    produce Spans which contain information that can be used to enrich the
    Humanloop File Kernels.

    For example, Instrumentors for LLM provider libraries intercept
    hyperparameters used in the API call to the model to build the
    Prompt File definition when using the @prompt decorator.

    Spans created that are not created by Humanloop decorators, such as
    those created by the Instrumentors mentioned above, will be passed
    to the Exporter as they are.
    """

    def __init__(
        self,
        exporter: SpanExporter,
        client: "BaseHumanloop",
    ) -> None:
        super().__init__(exporter)
        # span parent to span children map
        self._dependencies: dict[int, list[DependantSpan]] = defaultdict(list)
        self._waiting: dict[int, ReadableSpan] = {}
        # List of all span IDs that are contained in a Flow trace
        # They are passed to the Exporter as a span attribute
        # so the Exporter knows when to complete a trace
        self._spans_to_complete_flow_trace: dict[int, list[int]] = {}
        self._client = client

    def shutdown(self):
        return super().shutdown()

    def on_start(self, span, parent_context=None):
        """Hook executed on Span creation.

        Used for two cases:
            1. Complete the Flow trace after all Logs inside have been uploaded. The Flow trace
                spans are created bottoms-up. By the time the Flow span reaches the on_end hook,
                all spans inside have been passed to the Exporter. We attach the list of span IDs
                to the Flow span as an attribute, so the Exporter knows what spans (Logs) must
                be uploaded before the Flow trace is completed
            2. Instrument streaming Prompt decorated functions. The Instrumentor span will end only
                when the ChunksResponse is consumed, while the Prompt-decorated span will end when
                the function returns.
        """
        self._track_flow_traces(span)
        self._add_dependency_to_await(span)

    def on_end(self, span: ReadableSpan) -> None:
        span_id = span.context.span_id
        if is_humanloop_span(span=span):
            if not self._must_wait(span):
                self._send_to_exporter(
                    span=span,
                    dependencies=[dependency.span for dependency in self._dependencies[span.context.span_id]],
                )
            else:
                # Must wait for dependencies
                self._waiting[span_id] = span
            return

        if self._is_dependency(span):
            self._mark_dependency_arrival(span)
            self._send_to_exporter(span, [])

            waiting_span = self._get_waiting_parent(span)
            if waiting_span is not None:
                self._send_to_exporter(
                    span=span,
                    dependencies=[dependency.span for dependency in self._dependencies[span.context.span_id]],
                )
            return

        # Be unopinionated and pass all other spans to Exporter
        self._send_to_exporter(span=span, dependencies=[])

    def _must_wait(self, span: ReadableSpan) -> bool:
        if span.context.span_id not in self._dependencies:
            return False
        if all([dependency.finished for dependency in self._dependencies[span.context.span_id]]):
            return False
        return True

    def _get_waiting_parent(self, span: ReadableSpan) -> Optional[ReadableSpan]:
        # We know this span has a parent, need to satisfy the type checker
        parent_span_id = span.parent.span_id  # type: ignore
        if parent_span_id in self._waiting:
            if all([dependency.finished for dependency in self._dependencies[parent_span_id]]):
                waiting_span = self._waiting[parent_span_id]
                del self._dependencies[parent_span_id]
                del self._waiting[parent_span_id]
                return waiting_span
        return None

    def _add_dependency_to_await(self, span: ReadableSpan):
        # We know this span has a parent, need to satisfy the type checker
        if self._is_dependency(span):
            parent_span_id = span.parent.span_id  # type: ignore
            self._dependencies[parent_span_id].append(DependantSpan(span=span, finished=False))

    def _track_flow_traces(self, span: ReadableSpan):
        span_id = span.context.span_id

        if span.name == "humanloop.flow":
            # Head of a trace
            self._spans_to_complete_flow_trace[span_id] = []

        parent_span_id = span.parent.span_id if span.parent else None
        if parent_span_id and is_humanloop_span(span):
            # Log belongs to a trace, keep track of it
            for trace_head, all_trace_nodes in self._spans_to_complete_flow_trace.items():
                if parent_span_id == trace_head or parent_span_id in all_trace_nodes:
                    all_trace_nodes.append(span_id)
                    break

    def _mark_dependency_arrival(self, span: ReadableSpan):
        span_id = span.context.span_id
        # We know this span has a parent, need to satisfy type checker
        parent_span_id = span.parent.span_id  # type: ignore
        self._dependencies[parent_span_id] = [
            dependency if dependency.span.context.span_id != span_id else DependantSpan(span=span, finished=True)
            for dependency in self._dependencies[parent_span_id]
        ]

    def _send_to_exporter(
        self,
        span: ReadableSpan,
        dependencies: list[ReadableSpan],
    ):
        """
        Write attributes to the Humanloop spans depending on their type
        """

        if is_humanloop_span(span):
            # Processing common to all Humanloop File types
            self._write_start_end_times(span=span)

            # Processing specific to each Humanloop File type
            file_type = span.attributes[HUMANLOOP_FILE_TYPE_KEY]  # type: ignore
            span_id = span.context.span_id
            if file_type == "prompt":
                enhance_prompt_span(
                    client=self._client,
                    prompt_span=span,
                    dependencies=dependencies,
                )
            elif file_type == "tool":
                # No extra processing needed
                pass
            elif file_type == "flow":
                trace = self._spans_to_complete_flow_trace.get(span_id, [])
                write_to_opentelemetry_span(
                    span=span,
                    key=HUMANLOOP_FLOW_PREREQUISITES_KEY,
                    value=trace,
                )
            else:
                logger.error(
                    "[HumanloopSpanProcessor] Unknown Humanloop File span %s %s",
                    span_id,
                    span.name,
                )

        self.span_exporter.export([span])

        # Cleanup
        span_id = span.context.span_id
        if span_id in self._waiting:
            del self._waiting[span_id]
        if span_id in self._dependencies:
            del self._dependencies[span_id]
        if span_id in self._spans_to_complete_flow_trace:
            del self._spans_to_complete_flow_trace[span_id]

    @classmethod
    def _is_dependency(cls, span: ReadableSpan) -> bool:
        """Determine if the span contains information of interest for Spans created by Humanloop decorators."""
        # At the moment we only enrich Spans created by the Prompt decorators
        # As we add Instrumentors for other libraries, this function must
        # be expanded
        return span.parent is not None and (
            is_llm_provider_call(span=span) or span.name == HUMANLOOP_INTERCEPTED_HL_CALL_SPAN_NAME
        )

    @classmethod
    def _write_start_end_times(cls, span: ReadableSpan):
        if span.start_time:
            # NOTE: write_to_otel_span and read_from_otel_span have extra behavior
            # OTEL canonical way to write keys is to use the dot notation, as below
            # The 2 utilities encapsulate this behavior, allowing the dev to write
            # complex objects.
            # See doc-strings in humanloop.otel.helpers for more information
            span._attributes[f"{HUMANLOOP_LOG_KEY}.start_time"] = span.start_time / 1e9  # type: ignore
        if span.end_time:
            span._attributes[f"{HUMANLOOP_LOG_KEY}.end_time"] = span.end_time / 1e9  # type: ignore
            span._attributes[f"{HUMANLOOP_LOG_KEY}.created_at"] = span.end_time / 1e9  # type: ignore
