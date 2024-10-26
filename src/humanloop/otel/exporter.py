from queue import Queue
from threading import Thread
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

    WORK_THREADS = 8

    def __init__(self, client: "BaseHumanloop") -> None:
        super().__init__()
        self._client = client
        self._uploaded_log_ids = {}
        self._upload_queue = Queue()
        self._threads = [Thread(target=self._do_work, daemon=True) for _ in range(self.WORK_THREADS)]
        self._shutdown = False
        for thread in self._threads:
            thread.start()

    def export(self, spans: trace.Sequence[Span]) -> SpanExportResult:
        for span in spans:
            self._upload_queue.put(span)

    def shutdown(self) -> None:
        self._shutdown = True
        for thread in self._threads:
            thread.join()

    def force_flush(self, timeout_millis: int = 3000) -> bool:
        self._shutdown = True
        for thread in self._threads:
            thread.join(timeout=timeout_millis)
        self._upload_queue.join()

        return True

    def _do_work(self):
        # Do work while the Exporter was not instructed to
        # wind down or the queue is not empty
        while self._upload_queue.qsize() > 0 or not self._shutdown:
            try:
                # Don't block or the thread will never see the shutdown
                # command and will get stuck
                span_to_export: Span = self._upload_queue.get(block=False)
            except Exception:
                continue
            try:
                trace_metadata = read_from_opentelemetry_span(span_to_export, key=HL_TRACE_METADATA_KEY)
            except KeyError:
                trace_metadata = None
            if "trace_parent_id" not in trace_metadata or trace_metadata["trace_parent_id"] in self._uploaded_log_ids:
                # The Span is outside a Trace context or its parent has been uploaded
                # we can safely upload it to Humanloop
                self._export_dispatch(span_to_export)
            else:  # The parent has not been uploaded yet
                # Requeue the Span to be uploaded later
                self._upload_queue.put(span_to_export)
            self._upload_queue.task_done()

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
