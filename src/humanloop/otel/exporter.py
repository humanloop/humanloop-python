import logging
import typing
from queue import Queue
from threading import Thread
from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from humanloop.otel.constants import HL_FILE_OT_KEY, HL_LOG_OT_KEY, HL_OT_EMPTY_VALUE, HL_TRACE_METADATA_KEY
from humanloop.otel.helpers import is_humanloop_span, read_from_opentelemetry_span
from humanloop.requests.flow_kernel_request import FlowKernelRequestParams
from humanloop.requests.prompt_kernel_request import PromptKernelRequestParams

if typing.TYPE_CHECKING:
    from humanloop.base_client import BaseHumanloop


logger = logging.getLogger("humanloop.sdk")


class HumanloopSpanExporter(SpanExporter):
    """Upload Spans created by SDK decorators to Humanloop.

    Spans not created by Humanloop SDK decorators will be ignored.
    """

    DEFAULT_NUMBER_THREADS = 4

    def __init__(
        self,
        client: "BaseHumanloop",
        worker_threads: Optional[int] = None,
    ) -> None:
        """Upload Spans created by SDK decorators to Humanloop.

        Spans not created by Humanloop SDK decorators will be ignored.
        """
        super().__init__()
        self._client = client
        self._uploaded_log_ids: dict[
            str, str
        ] = {}  # Uploaded spans translate to a Log on Humanloop. The IDs are required to link Logs in a Flow Trace
        self._upload_queue: Queue = Queue()  # Work queue for the threads uploading the spans
        self._threads: list[Thread] = [
            Thread(target=self._do_work, daemon=True) for _ in range(worker_threads or self.DEFAULT_NUMBER_THREADS)
        ]
        self._shutdown: bool = (
            False  # Signals threads no more work will arrive and they should wind down if the queue is empty
        )
        for thread in self._threads:
            thread.start()

    def export(self, spans: trace.Sequence[ReadableSpan]) -> SpanExportResult:
        if not self._shutdown:
            for span in spans:
                if is_humanloop_span(span):
                    self._upload_queue.put(span)
            return SpanExportResult.SUCCESS
        else:
            logger.warning("HumanloopSpanExporter is shutting down, not accepting new spans")
            return SpanExportResult.FAILURE

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
        """Upload spans to Humanloop.

        Ran by worker threads. The threads use the self._shutdown flag to wait
        for Spans to arrive. Setting a timeout on self._upload_queue.get() risks
        shutting down the thread early as no Spans are produced e.g. while waiting
        for user input into the instrumented feature or application.

        Each thread will upload a Span to Humanloop, provided the Span has all its
        dependencies uploaded. The dependency happens in a Flow Trace context, where
        the Trace parent must be uploaded first. The Span Processor will send in Spans
        bottoms-up, while the upload of a Trace happens top-down. If a Span did not
        have its span uploaded yet, it will be requeued to be uploaded later.
        """
        # Do work while the Exporter was not instructed to
        # wind down or the queue is not empty
        while self._upload_queue.qsize() > 0 or not self._shutdown:
            try:
                # Don't block or the thread will never see the shutdown
                # command and will get stuck
                span_to_export = self._upload_queue.get(block=False)
            except Exception:
                continue
            try:
                trace_metadata = read_from_opentelemetry_span(
                    span_to_export,
                    key=HL_TRACE_METADATA_KEY,
                )
            except KeyError:
                trace_metadata = None
            if "trace_parent_id" not in trace_metadata or trace_metadata["trace_parent_id"] in self._uploaded_log_ids:
                # The Span is outside a Trace context or its parent has been uploaded
                # we can safely upload it to Humanloop
                self._export_span_dispatch(span_to_export)
            else:  # The parent has not been uploaded yet
                # Requeue the Span to be uploaded later
                self._upload_queue.put(span_to_export)
            self._upload_queue.task_done()

    def _export_span_dispatch(self, span: ReadableSpan) -> None:
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

    def _export_prompt(self, span: ReadableSpan) -> None:
        file_object: dict[str, Any] = read_from_opentelemetry_span(
            span,
            key=HL_FILE_OT_KEY,
        )
        log_object: dict[str, Any] = read_from_opentelemetry_span(
            span,
            key=HL_LOG_OT_KEY,
        )
        # NOTE: Due to Otel conventions, attributes with value of None are removed
        # If not present, instantiate as empty dictionary
        if "inputs" not in log_object:
            log_object["inputs"] = {}
        # NOTE: Due to Otel conventions, lists are read as dictionaries
        # E.g. ["a", "b"] -> {"0": "a", "1": "b"}
        # We must convert the dictionary back to a list
        # See humanloop.otel.helpers._list_to_ott
        if "messages" not in log_object:
            log_object["messages"] = []
        else:
            log_object["messages"] = list(log_object["messages"].values())
        trace_metadata: Optional[dict[str, str]]
        try:
            trace_metadata = read_from_opentelemetry_span(
                span,
                key=HL_TRACE_METADATA_KEY,
            )  # type: ignore
        except KeyError:
            trace_metadata = None
        if trace_metadata:
            trace_parent_id = self._uploaded_log_ids[trace_metadata["trace_parent_id"]]
        else:
            trace_parent_id = None
        prompt: PromptKernelRequestParams = file_object["prompt"]
        path: str = file_object["path"]
        response = self._client.prompts.log(
            path=path,
            prompt=prompt,
            **log_object,
            trace_parent_id=trace_parent_id,
        )
        self._uploaded_log_ids[span.context.span_id] = response.id

    def _export_tool(self, span: ReadableSpan) -> None:
        file_object: dict[str, Any] = read_from_opentelemetry_span(span, key=HL_FILE_OT_KEY)
        log_object: dict[str, Any] = read_from_opentelemetry_span(span, key=HL_LOG_OT_KEY)
        trace_metadata: Optional[dict[str, str]]
        try:
            # HL_TRACE_METADATA_KEY is a dict[str, str], has no nesting
            trace_metadata = read_from_opentelemetry_span(span, key=HL_TRACE_METADATA_KEY)  # type: ignore
        except KeyError:
            trace_metadata = None
        if trace_metadata:
            trace_parent_id = self._uploaded_log_ids[trace_metadata["trace_parent_id"]]
        else:
            trace_parent_id = None
        tool = file_object["tool"]
        if tool.get("attributes", HL_OT_EMPTY_VALUE) == HL_OT_EMPTY_VALUE:
            tool["attributes"] = {}
        if tool.get("setup_values", HL_OT_EMPTY_VALUE) == HL_OT_EMPTY_VALUE:
            tool["setup_values"] = {}
        path: str = file_object["path"]
        response = self._client.tools.log(
            path=path,
            tool=tool,
            **log_object,
            trace_parent_id=trace_parent_id,
        )
        self._uploaded_log_ids[span.context.span_id] = response.id

    def _export_flow(self, span: ReadableSpan) -> None:
        file_object: dict[str, Any] = read_from_opentelemetry_span(span, key=HL_FILE_OT_KEY)
        log_object: dict[str, Any] = read_from_opentelemetry_span(span, key=HL_LOG_OT_KEY)
        trace_metadata: Optional[dict[str, str]]
        try:
            trace_metadata = read_from_opentelemetry_span(span, key=HL_TRACE_METADATA_KEY)  # type: ignore
        except KeyError:
            trace_metadata = None
        if trace_metadata and "trace_parent_id" in trace_metadata:
            trace_parent_id = self._uploaded_log_ids[trace_metadata["trace_parent_id"]]
        else:
            trace_parent_id = None
        # Cannot write falsy values except None in OTel Span attributes
        # If a None write is attempted then the attribute is removed
        # making it impossible to distinguish between a Flow Span and
        # Spans not created by Humanloop (see humanloop.otel.helpers.is_humanloop_span)
        flow: FlowKernelRequestParams
        if file_object["flow"] == HL_OT_EMPTY_VALUE:
            flow = {"attributes": {}}
        else:
            flow = file_object["flow"]
        path: str = file_object["path"]
        response = self._client.flows.log(
            path=path,
            flow=flow,
            **log_object,
            trace_parent_id=trace_parent_id,
        )
        self._uploaded_log_ids[span.context.span_id] = response.id
