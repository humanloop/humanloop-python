import contextvars
import logging
import threading
import time
import typing
from queue import Empty as EmptyQueue
from queue import Queue
from threading import Thread
from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from humanloop.core import ApiError as HumanloopApiError
from humanloop.eval_utils.context import EVALUATION_CONTEXT_VARIABLE_NAME, EvaluationContext
from humanloop.otel.constants import (
    HUMANLOOP_FILE_KEY,
    HUMANLOOP_FILE_TYPE_KEY,
    HUMANLOOP_FLOW_PREREQUISITES_KEY,
    HUMANLOOP_LOG_KEY,
    HUMANLOOP_PATH_KEY,
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

    Spans not created by Humanloop SDK decorators will be ignored.
    """

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
        self._span_id_to_uploaded_log_id: dict[int, Optional[str]] = {}
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
        # they should wind down if the queue is empty
        self._shutdown: bool = False
        for thread in self._threads:
            thread.start()
            logger.debug("Exporter Thread %s started", thread.ident)
        # Flow Log Span ID mapping to children Spans that must be uploaded first
        self._flow_log_prerequisites: dict[int, set[int]] = {}

    def export(self, spans: trace.Sequence[ReadableSpan]) -> SpanExportResult:
        def is_evaluated_file(
            span: ReadableSpan,
            evaluation_context: Optional[EvaluationContext],
        ) -> bool:
            if evaluation_context is None:
                return False

            return span.attributes.get(HUMANLOOP_PATH_KEY) == evaluation_context["path"]  # type: ignore

        if not self._shutdown:
            try:
                evaluation_context = self._client.evaluation_context_variable.get()
                if len(spans) > 1:
                    raise RuntimeError("HumanloopSpanExporter expected a single span when running an evaluation")
                if not is_evaluated_file(spans[0], evaluation_context):
                    evaluation_context = None
            except LookupError:
                # No ongoing Evaluation happening
                evaluation_context = None
            for span in spans:
                if is_humanloop_span(span):
                    # We pass the EvaluationContext from the eval_run utility thread to
                    # the export thread so the .log action works as expected
                    evaluation_context_copy = None
                    for context_var, context_var_value in contextvars.copy_context().items():
                        if context_var.name == EVALUATION_CONTEXT_VARIABLE_NAME:
                            evaluation_context_copy = context_var_value
                    self._upload_queue.put(
                        (
                            span,
                            evaluation_context_copy,
                        ),
                    )
                    logger.debug(
                        "[HumanloopSpanExporter] Span %s %s with EvaluationContext %s added to upload queue",
                        span.context.span_id,
                        span.name,
                        evaluation_context_copy,
                    )
            # Reset the EvaluationContext so run eval does not
            # create a duplicate Log
            if evaluation_context is not None and is_evaluated_file(
                spans[0],
                evaluation_context,
            ):
                logger.debug(
                    "[HumanloopSpanExporter] EvaluationContext %s marked as exhausted for Log in Span %s",
                    evaluation_context,
                    spans[0].attributes,
                )
                # Mark the EvaluationContext as used
                self._client.evaluation_context_variable.set(None)
            return SpanExportResult.SUCCESS
        else:
            logger.warning("[HumanloopSpanExporter] Shutting down, not accepting new spans")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self._shutdown = True
        for thread in self._threads:
            thread.join()
            logger.debug("[HumanloopSpanExporter] Exporter Thread %s joined", thread.ident)

    def force_flush(self, timeout_millis: int = 10000) -> bool:
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
        have its span uploaded yet, it will be re-queued to be uploaded later.
        """

        # Do work while the Exporter was not instructed to
        # wind down or the queue is not empty
        while self._upload_queue.qsize() > 0 or not self._shutdown:
            try:
                thread_args: tuple[ReadableSpan, EvaluationContext]  # type: ignore
                # Don't block or the thread will never be notified of the shutdown
                thread_args = self._upload_queue.get(
                    block=False,
                )  # type: ignore
                span_to_export, evaluation_context = thread_args
                # Set the EvaluationContext for the thread so the .log action works as expected
                # NOTE: Expecting the evaluation thread to send a single span so we are
                #   not resetting the EvaluationContext in the scope of the export thread
                self._client.evaluation_context_variable.set(evaluation_context)
            except EmptyQueue:
                continue
            if span_to_export.parent is None:
                # Span is not part of a Flow Log
                self._export_span_dispatch(span_to_export)
                logger.debug(
                    "[HumanloopSpanExporter] _do_work on Thread %s: Dispatching span %s %s",
                    threading.get_ident(),
                    span_to_export.context.span_id,
                    span_to_export.name,
                )
            elif span_to_export.parent.span_id in self._span_id_to_uploaded_log_id:
                # Span is part of a Flow and its parent has been uploaded
                self._export_span_dispatch(span_to_export)
                logger.debug(
                    "[HumanloopSpanExporter] _do_work on Thread %s: Dispatching span %s %s",
                    threading.get_ident(),
                    span_to_export.context.span_id,
                    span_to_export.name,
                )
            else:
                # Requeue the Span and upload after its parent
                self._upload_queue.put((span_to_export, evaluation_context))
            self._upload_queue.task_done()

    def _mark_span_completed(self, span_id: int) -> None:
        for flow_log_span_id, flow_children_span_ids in self._flow_log_prerequisites.items():
            if span_id in flow_children_span_ids:
                flow_children_span_ids.remove(span_id)
                if len(flow_children_span_ids) == 0:
                    # All logs in the Trace have been uploaded, mark the Flow Log as complete
                    flow_log_id = self._span_id_to_uploaded_log_id[flow_log_span_id]
                    if flow_log_id is None:
                        logger.error(
                            "[HumanloopSpanExporter] Cannot complete Flow log %s, log ID is None",
                            flow_log_span_id,
                        )
                    else:
                        self._client.flows.update_log(log_id=flow_log_id, trace_status="complete")
                break

    def _export_span_dispatch(self, span: ReadableSpan) -> None:
        hl_file = read_from_opentelemetry_span(span, key=HUMANLOOP_FILE_KEY)
        file_type = span._attributes.get(HUMANLOOP_FILE_TYPE_KEY)  # type: ignore
        parent_span_id = span.parent.span_id if span.parent else None

        while parent_span_id and self._span_id_to_uploaded_log_id.get(parent_span_id) is None:
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
            export_func = self._export_prompt
        elif file_type == "tool":
            export_func = self._export_tool
        elif file_type == "flow":
            export_func = self._export_flow
        else:
            raise NotImplementedError(f"Unknown span type: {hl_file}")
        export_func(span=span)

    def _export_prompt(self, span: ReadableSpan) -> None:
        file_object: dict[str, Any] = read_from_opentelemetry_span(
            span,
            key=HUMANLOOP_FILE_KEY,
        )
        log_object: dict[str, Any] = read_from_opentelemetry_span(
            span,
            key=HUMANLOOP_LOG_KEY,
        )
        # NOTE: Due to OTel conventions, attributes with value of None are removed
        # If not present, instantiate as empty dictionary
        if "inputs" not in log_object:
            log_object["inputs"] = {}
        if "messages" not in log_object:
            log_object["messages"] = []
        if "tools" not in file_object["prompt"]:
            file_object["prompt"]["tools"] = []

        path: str = file_object["path"]
        prompt: PromptKernelRequestParams = file_object["prompt"]

        span_parent_id = span.parent.span_id if span.parent else None
        trace_parent_id = self._span_id_to_uploaded_log_id[span_parent_id] if span_parent_id else None

        if "attributes" not in prompt or not prompt["attributes"]:
            prompt["attributes"] = {}

        try:
            log_response = self._client.prompts.log(
                path=path,
                prompt=prompt,
                **log_object,
                trace_parent_id=trace_parent_id,
            )
            self._span_id_to_uploaded_log_id[span.context.span_id] = log_response.id
        except HumanloopApiError:
            self._span_id_to_uploaded_log_id[span.context.span_id] = None
        self._mark_span_completed(span_id=span.context.span_id)

    def _export_tool(self, span: ReadableSpan) -> None:
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

        span_parent_id = span.parent.span_id if span.parent else None
        trace_parent_id = self._span_id_to_uploaded_log_id[span_parent_id] if span_parent_id else None

        # API expects an empty dictionary if user does not supply attributes
        if not tool.get("attributes"):
            tool["attributes"] = {}
        if not tool.get("setup_values"):
            tool["setup_values"] = {}
        if "parameters" in tool["function"] and "properties" not in tool["function"]["parameters"]:
            tool["function"]["parameters"]["properties"] = {}

        try:
            log_response = self._client.tools.log(
                path=path,
                tool=tool,
                **log_object,
                trace_parent_id=trace_parent_id,
            )
            self._span_id_to_uploaded_log_id[span.context.span_id] = log_response.id
        except HumanloopApiError:
            self._span_id_to_uploaded_log_id[span.context.span_id] = None
        self._mark_span_completed(span_id=span.context.span_id)

    def _export_flow(self, span: ReadableSpan) -> None:
        file_object: dict[str, Any] = read_from_opentelemetry_span(
            span,
            key=HUMANLOOP_FILE_KEY,
        )
        log_object: dict[str, Any] = read_from_opentelemetry_span(
            span,
            key=HUMANLOOP_LOG_KEY,
        )
        # Spans that must be uploaded before the Flow Span is completed
        try:
            prerequisites: list[int] = read_from_opentelemetry_span(  # type: ignore
                span=span,
                key=HUMANLOOP_FLOW_PREREQUISITES_KEY,
            )
            self._flow_log_prerequisites[span.context.span_id] = set(prerequisites)
        except KeyError:
            self._flow_log_prerequisites[span.context.span_id] = set()

        path: str = file_object["path"]
        flow: FlowKernelRequestParams
        if not file_object.get("flow"):
            flow = {"attributes": {}}
        else:
            flow = file_object["flow"]

        span_parent_id = span.parent.span_id if span.parent else None
        trace_parent_id = self._span_id_to_uploaded_log_id[span_parent_id] if span_parent_id else None

        if "output" not in log_object:
            log_object["output"] = None
        try:
            log_response = self._client.flows.log(
                path=path,
                flow=flow,
                **log_object,
                trace_parent_id=trace_parent_id,
            )
            self._span_id_to_uploaded_log_id[span.get_span_context().span_id] = log_response.id
        except HumanloopApiError as e:
            logger.error(str(e))
            self._span_id_to_uploaded_log_id[span.context.span_id] = None
        self._mark_span_completed(span_id=span.context.span_id)
