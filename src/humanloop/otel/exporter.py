import logging
import threading

import typing
from queue import Empty as EmptyQueue
from queue import Queue
from threading import Thread
from typing import Any, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from humanloop.core import ApiError as HumanloopApiError
from humanloop.eval_utils.context import (
    EvaluationContext,
    evaluation_context_set,
    get_evaluation_context,
    set_evaluation_context,
)
from humanloop.otel.constants import (
    HUMANLOOP_FILE_KEY,
    HUMANLOOP_FILE_TYPE_KEY,
    HUMANLOOP_FLOW_PREREQUISITES_KEY,
    HUMANLOOP_LOG_KEY,
)
from humanloop.otel.helpers import is_humanloop_span, read_from_opentelemetry_span
from humanloop.requests.flow_kernel_request import FlowKernelRequestParams
from humanloop.requests.prompt_kernel_request import PromptKernelRequestParams
from humanloop.requests.tool_kernel_request import ToolKernelRequestParams

if typing.TYPE_CHECKING:
    from humanloop.client import Humanloop


logger = logging.getLogger("humanloop.sdk")


class HumanloopSpanExporter(SpanExporter):
    """Upload Spans created by SDK decorators to Humanloop.

    Spans not created by Humanloop SDK decorators will be dropped.

    Each Humanloop Span contains information about the File to log against and
    the Log to create. We are using the .log actions that pass the kernel in the
    request. This allows us to create new Versions if the decorated function
    is changed.

    The exporter uploads Spans top-to-bottom, where a Span is uploaded only after
    its parent Span has been uploaded. This is necessary for Flow Traces, where
    the parent Span is a Flow Log and the children are the Logs in the Trace.

    The exporter keeps an upload queue and only uploads a Span if its direct parent has
    been uploaded.
    """

    # NOTE: LLM Instrumentors will only intercept calls to the provider made via the
    # official libraries e.g. import openai from openai. This is 100% the reason why
    # prompt call is not intercepted by the Instrumentor. The way to fix this is likely
    # overriding the hl_client.prompt.call utility. @James I'll do this since it will
    # involve looking at the EvaluationContext deep magic.

    DEFAULT_NUMBER_THREADS = 4

    def __init__(
        self,
        client: "Humanloop",
        worker_threads: Optional[int] = None,
    ) -> None:
        """Upload Spans created by SDK decorators to Humanloop.

        Spans not created by Humanloop SDK decorators will be ignored.
        """
        super().__init__()
        self._client = client
        # Uploaded spans translate to a Log on Humanloop. The IDs are required to link Logs in a Flow Trace
        self._span_to_uploaded_log_id: dict[int, Optional[str]] = {}
        # Work queue for the threads uploading the spans
        self._upload_queue: Queue = Queue()
        # Worker threads to export the spans
        self._threads: list[Thread] = [
            Thread(
                target=self._do_work,
                daemon=True,
            )
            for _ in range(worker_threads or self.DEFAULT_NUMBER_THREADS)
        ]
        # Signals threads no more work will arrive and
        # they should wind down after they empty the queue
        self._shutdown: bool = False
        # Init the upload threads
        for thread in self._threads:
            thread.start()
            logger.debug("Exporter Thread %s started", thread.ident)
        # Flow Log Span ID mapping to children Spans that must be uploaded first
        self._spans_left_in_trace: dict[int, set[int]] = {}
        self._traces: list[set[str]] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._shutdown:
            logger.warning("[HumanloopSpanExporter] Shutting down, not accepting new spans")
            return SpanExportResult.FAILURE

        for span in spans:
            if not is_humanloop_span(span):
                continue

            self._upload_queue.put(
                (
                    span,
                    get_evaluation_context() if evaluation_context_set() else None,
                ),
            )

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._shutdown = True
        for thread in self._threads:
            thread.join(timeout=5)
            logger.debug("[HumanloopSpanExporter] Exporter Thread %s joined", thread.ident)

    def force_flush(self, timeout_millis: int = 5000) -> bool:
        self._shutdown = True
        for thread in self._threads:
            thread.join(timeout=timeout_millis / 1000)
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
        have its span uploaded yet, it will be re-queued to be uploaded later.
        """

        # Do work while the Exporter was not instructed to
        # wind down or the queue is not empty
        while self._upload_queue.qsize() > 0 or not self._shutdown:
            thread_args: tuple[ReadableSpan, Optional[EvaluationContext]]  # type: ignore
            try:
                # Don't block or the thread will never be notified of the shutdown
                thread_args = self._upload_queue.get(
                    block=False,
                )  # type: ignore
            except EmptyQueue:
                # Wait for the another span to arrive
                continue

            span_to_export, evaluation_context = thread_args
            if evaluation_context is not None:
                # Context variables are thread scoped
                # One existed in the eval_run utility thread
                # so it must be copied over to the current
                # exporter thread
                set_evaluation_context(evaluation_context)

            if span_to_export.parent is None:
                # Span cannot be part of a Flow trace
                self._export_span_dispatch(span_to_export)
                logger.debug(
                    "[HumanloopSpanExporter] _do_work on Thread %s: Dispatching span %s %s",
                    threading.get_ident(),
                    span_to_export.context.span_id,
                    span_to_export.name,
                )

            elif span_to_export.parent.span_id in self._span_to_uploaded_log_id:
                # Span is part of a Flow trace and its parent has been uploaded
                self._export_span_dispatch(span_to_export)

            else:
                # Requeue the Span and upload after its parent
                self._upload_queue.put((span_to_export, evaluation_context))

            # Notify the shared queue that we are done
            # with the current head of the task queue
            self._upload_queue.task_done()

    def _export_span_dispatch(self, span: ReadableSpan) -> None:
        """Call the appropriate BaseHumanloop.X.log based on the Span type."""
        file_type = span._attributes.get(HUMANLOOP_FILE_TYPE_KEY)  # type: ignore
        parent_span_id = span.parent.span_id if span.parent else None

        while parent_span_id and self._span_to_uploaded_log_id.get(parent_span_id) is None:
            logger.debug(
                "[HumanloopSpanExporter] _export_span_dispatch on Thread %s Span %s %s waiting for parent %s to be uploaded",
                threading.get_ident(),
                span.context.span_id,
                span.name,
                parent_span_id,
            )

        logger.debug(
            "[HumanloopSpanExporter] Exporting span %s with file type %s",
            span,
            file_type,
        )

        if file_type == "prompt":
            self._export_prompt_span(span=span)
        elif file_type == "tool":
            self._export_tool_span(span=span)
        elif file_type == "flow":
            self._export_flow_span(span=span)
        else:
            raise NotImplementedError(f"Unknown span type: {file_type}")

    def _export_prompt_span(self, span: ReadableSpan) -> None:
        file_object: dict[str, Any] = read_from_opentelemetry_span(
            span,
            key=HUMANLOOP_FILE_KEY,
        )
        log_object: dict[str, Any] = read_from_opentelemetry_span(
            span,
            key=HUMANLOOP_LOG_KEY,
        )
        # NOTE: Due to OTEL conventions, attributes with value of None are removed
        # on write to Span. If not present, instantiate these as empty
        if "inputs" not in log_object:
            log_object["inputs"] = {}
        if "messages" not in log_object:
            log_object["messages"] = []
        if "tools" not in file_object["prompt"]:
            file_object["prompt"]["tools"] = []

        path: str = file_object["path"]
        prompt: PromptKernelRequestParams = file_object["prompt"]

        trace_parent_id = self._get_parent_in_trace(span)

        if "attributes" not in prompt or not prompt["attributes"]:
            prompt["attributes"] = {}

        try:
            log_response = self._client.prompts.log(
                path=path,
                prompt=prompt,
                **log_object,
                trace_parent_id=trace_parent_id,
            )
            self._span_to_uploaded_log_id[span.context.span_id] = log_response.id
            if trace_parent_id is not None:
                self._keep_track_of_trace(log_response.id, trace_parent_id)
        except HumanloopApiError:
            self._span_to_uploaded_log_id[span.context.span_id] = None
        self._mark_span_as_uploaded(span_id=span.context.span_id)

    def _export_tool_span(self, span: ReadableSpan) -> None:
        file_object: dict[str, Any] = read_from_opentelemetry_span(
            span,
            key=HUMANLOOP_FILE_KEY,
        )
        log_object: dict[str, Any] = read_from_opentelemetry_span(
            span,
            key=HUMANLOOP_LOG_KEY,
        )

        path: str = file_object["path"]
        tool: ToolKernelRequestParams = file_object["tool"]

        # API expects an empty dictionary if user does not supply attributes
        # NOTE: see comment in _export_prompt_span about OTEL conventions
        if not tool.get("attributes"):
            tool["attributes"] = {}
        if not tool.get("setup_values"):
            tool["setup_values"] = {}
        if "parameters" in tool["function"] and "properties" not in tool["function"]["parameters"]:
            tool["function"]["parameters"]["properties"] = {}

        trace_parent_id = self._get_parent_in_trace(span)
        try:
            log_response = self._client.tools.log(
                path=path,
                tool=tool,
                **log_object,
                trace_parent_id=trace_parent_id,
            )
            self._span_to_uploaded_log_id[span.context.span_id] = log_response.id
            if trace_parent_id is not None:
                self._keep_track_of_trace(log_response.id, trace_parent_id)
        except HumanloopApiError:
            self._span_to_uploaded_log_id[span.context.span_id] = None
        self._mark_span_as_uploaded(span_id=span.context.span_id)

    def _export_flow_span(self, span: ReadableSpan) -> None:
        file_object: dict[str, Any] = read_from_opentelemetry_span(
            span,
            key=HUMANLOOP_FILE_KEY,
        )
        log_object: dict[str, Any] = read_from_opentelemetry_span(
            span,
            key=HUMANLOOP_LOG_KEY,
        )
        # Spans that must be uploaded before the Flow Span is completed
        # We instantiate the list of prerequisites from the attribute
        # passed by the Processor. Each uploaded child in the trace
        # will check if it's the last one and mark the Flow Log as complete
        try:
            prerequisites: list[int] = read_from_opentelemetry_span(  # type: ignore
                span=span,
                key=HUMANLOOP_FLOW_PREREQUISITES_KEY,
            )
            self._spans_left_in_trace[span.context.span_id] = set(prerequisites)
        except KeyError:
            # OTEL will drop falsy attributes, so if a Flow has no prerequisites
            # the attribute will not be present
            self._spans_left_in_trace[span.context.span_id] = set()

        path: str = file_object["path"]
        flow: FlowKernelRequestParams
        if not file_object.get("flow"):
            flow = {"attributes": {}}
        else:
            flow = file_object["flow"]

        trace_parent_id = self._get_parent_in_trace(span)

        if "output" not in log_object:
            log_object["output"] = None
        try:
            log_response = self._client.flows.log(
                path=path,
                flow=flow,
                **log_object,
                trace_parent_id=trace_parent_id,
            )
            if trace_parent_id is not None:
                self._keep_track_of_trace(
                    log_id=log_response.id,
                    parent_log_id=trace_parent_id,
                )
            # Exporting a flow log creates a new trace
            self._traces.append({log_response.id})
            self._span_to_uploaded_log_id[span.get_span_context().span_id] = log_response.id
        except HumanloopApiError as e:
            logger.error(str(e))
            self._span_to_uploaded_log_id[span.context.span_id] = None
        self._mark_span_as_uploaded(span_id=span.context.span_id)

    def _mark_span_as_uploaded(self, span_id: int) -> None:
        """Mark a Span as uploaded for Flow trace completion.

        If this Span corresponds to the last child in the Flow trace,
        mark the Flow Log as complete.
        """
        for trace_head_span_id, spans_left in self._spans_left_in_trace.items():
            if span_id in spans_left:
                spans_left.remove(span_id)
                self._mark_trace_complete_if_needed(trace_head_span_id=trace_head_span_id)
                # Found the trace the span belongs to
                # break from for loop
                break

    def _mark_trace_complete_if_needed(self, trace_head_span_id: int):
        spans_to_complete = self._spans_left_in_trace[trace_head_span_id]
        if len(spans_to_complete) == 0:
            flow_log_id = self._span_to_uploaded_log_id[trace_head_span_id]
            if flow_log_id is None:
                # Uploading the head of the Flow trace failed
                logger.error(
                    "[HumanloopSpanExporter] Cannot complete Flow log %s, log ID is None",
                    trace_head_span_id,
                )
            else:
                self._client.flows.update_log(log_id=flow_log_id, trace_status="complete")

    def _keep_track_of_trace(self, log_id: str, parent_log_id: str):
        found = False
        for trace in self._traces:
            if parent_log_id in trace:
                trace.add(log_id)
                found = True
            if found:
                break

    def _get_parent_in_trace(self, span: ReadableSpan) -> Optional[str]:
        if span.parent is None:
            return None
        parent_log_id = self._span_to_uploaded_log_id[span.parent.span_id]
        for trace in self._traces:
            if parent_log_id in trace:
                return parent_log_id
        return None
