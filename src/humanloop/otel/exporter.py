import typing
from opentelemetry import trace
from opentelemetry.sdk.trace import Span
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from humanloop.otel.constants import HL_FILE_OT_KEY, HL_LOG_OT_KEY, HL_TRACE_METADATA_KEY, OT_EMPTY_ATTRIBUTE
from humanloop.otel.helpers import read_from_opentelemetry_span

if typing.TYPE_CHECKING:
    from humanloop.base_client import BaseHumanloop


class HumanloopSpanExporter(SpanExporter):
    """SpanExporter that uploads OpenTelemetry spans to Humanloop Humanloop spans."""

    def __init__(self, client: "BaseHumanloop") -> None:
        super().__init__()
        self._client = client
        self._uploaded_log_ids = {}
        self._upload_queue = []

    def _export_prompt(self, span: Span) -> None:
        file_object = read_from_opentelemetry_span(span, key=HL_FILE_OT_KEY)
        log_object = read_from_opentelemetry_span(span, key=HL_LOG_OT_KEY)
        try:
            trace_metadata = read_from_opentelemetry_span(span, key=HL_TRACE_METADATA_KEY)

        except KeyError:
            trace_metadata = None
        if trace_metadata:
            trace_parent_id = self._uploaded_log_ids[trace_metadata["trace_parent_id"]]
        else:
            trace_parent_id = None
        prompt = file_object["prompt"]
        path = file_object["path"]
        response = self._client.prompts.log(
            path=path,
            prompt=prompt,
            **log_object,
            trace_parent_id=trace_parent_id,
        )
        self._uploaded_log_ids[span.context.span_id] = response.id

    def _export_tool(self, span: Span) -> None:
        file_object = read_from_opentelemetry_span(span, key=HL_FILE_OT_KEY)
        log_object = read_from_opentelemetry_span(span, key=HL_LOG_OT_KEY)
        try:
            trace_metadata = read_from_opentelemetry_span(span, key=HL_TRACE_METADATA_KEY)
        except KeyError:
            trace_metadata = None
        if trace_metadata:
            trace_parent_id = self._uploaded_log_ids[trace_metadata["trace_parent_id"]]
        else:
            trace_parent_id = None
        tool = file_object["tool"]
        path = file_object["path"]
        response = self._client.tools.log(
            path=path,
            tool=tool,
            **log_object,
            trace_parent_id=trace_parent_id,
        )
        self._uploaded_log_ids[span.context.span_id] = response.id

    def _export_flow(self, span: Span) -> None:
        file_object = read_from_opentelemetry_span(span, key=HL_FILE_OT_KEY)
        log_object = read_from_opentelemetry_span(span, key=HL_LOG_OT_KEY)
        try:
            trace_metadata = read_from_opentelemetry_span(span, key=HL_TRACE_METADATA_KEY)
        except KeyError:
            trace_metadata = None
        if trace_metadata and "trace_parent_id" in trace_metadata:
            trace_parent_id = self._uploaded_log_ids[trace_metadata["trace_parent_id"]]
        else:
            trace_parent_id = None
        flow = file_object["flow"]
        if flow == OT_EMPTY_ATTRIBUTE:
            flow = {
                "attributes": {},
            }
        path = file_object["path"]
        response = self._client.flows.log(
            path=path,
            flow=flow,
            **log_object,
            trace_parent_id=trace_parent_id,
        )
        self._uploaded_log_ids[span.context.span_id] = response.id

    def _export_dispatch(self, span: Span) -> None:
        hl_file = read_from_opentelemetry_span(span, key=HL_FILE_OT_KEY)

        if "prompt" in hl_file:
            export_func = self._export_prompt
        elif "tool" in hl_file:
            export_func = self._export_tool
        elif "flow" in hl_file:
            export_func = self._export_flow
        else:
            raise NotImplementedError(f"Unknown span type: {hl_file}")
        export_func(span=span)

    def export(self, spans: trace.Sequence[Span]) -> SpanExportResult:
        # TODO: Put this on a separate thread
        for span in spans:
            try:
                flow_metadata = read_from_opentelemetry_span(span, key=HL_TRACE_METADATA_KEY)
            except KeyError:
                flow_metadata = None
            if flow_metadata:
                # Span is part of a Flow, queue up Spans for upload until the Trace Head is exported.
                # The spans arrive at the Exporter in reverse order or creation, as they end.
                # We insert them at the front of the queue so that they are processed in the correct order
                self._upload_queue.insert(0, span)
                if (
                    flow_metadata["is_flow_log"]
                    # The Flow might be nested in another Flow
                    # i.e. has trace_parent_id set.
                    # Wait until the top level Flow is exported
                    and "trace_parent_id" not in flow_metadata
                ):
                    # TODO: Add threading to this: sibling Spans on the same
                    # depth level in the Trace can be uploaded in parallel
                    while len(self._upload_queue) > 0:
                        span = self._upload_queue.pop(0)
                        self._export_dispatch(span)
            else:
                # Span is not part of Flow, upload as singular
                self._export_dispatch(span)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        # TODO: When implementing the multi-threaded version of export, this will need to be updated
        return True
