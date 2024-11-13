import contextvars
import copy
import json
import logging
import threading
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
from humanloop.otel import TRACE_FLOW_CONTEXT, FlowContext
from humanloop.otel.constants import (
    HUMANLOOP_FILE_KEY,
    HUMANLOOP_FILE_TYPE_KEY,
    HUMANLOOP_LOG_KEY,
    HUMANLOOP_PATH_KEY,
)
from humanloop.otel.helpers import is_humanloop_span, read_from_opentelemetry_span
from humanloop.requests.flow_kernel_request import FlowKernelRequestParams
from humanloop.requests.prompt_kernel_request import PromptKernelRequestParams

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
                        "Span %s with EvaluationContext %s added to upload queue",
                        span.attributes,
                        evaluation_context_copy,
                    )
            # Reset the EvaluationContext so run eval does not
            # create a duplicate Log
            if evaluation_context is not None and is_evaluated_file(
                spans[0],
                evaluation_context,
            ):
                logger.debug(
                    "EvaluationContext %s marked as exhausted for Log in Span %s",
                    evaluation_context,
                    spans[0].attributes,
                )
                # Mark the EvaluationContext as used
                self._client.evaluation_context_variable.set(None)
            return SpanExportResult.SUCCESS
        else:
            logger.warning("HumanloopSpanExporter is shutting down, not accepting new spans")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self._shutdown = True
        for thread in self._threads:
            thread.join()
            logger.debug("Exporter Thread %s joined", thread.ident)

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
            trace_metadata = TRACE_FLOW_CONTEXT.get(span_to_export.get_span_context().span_id)
            if trace_metadata is None:
                # Span is not part of a Flow Log
                self._export_span_dispatch(span_to_export)
                logger.debug(
                    "_do_work on Thread %s: Dispatched span %s with FlowContext %s which is not part of a Flow",
                    threading.get_ident(),
                    span_to_export.attributes,
                    trace_metadata,
                )
            elif trace_metadata["trace_parent_id"] is None:
                # Span is the head of a Flow Trace
                self._export_span_dispatch(span_to_export)
                logger.debug(
                    "Dispatched span %s which is a Flow Log with FlowContext %s",
                    span_to_export.attributes,
                    trace_metadata,
                )
            elif trace_metadata["trace_parent_id"] in self._span_id_to_uploaded_log_id:
                # Span is part of a Flow and its parent has been uploaded
                self._export_span_dispatch(span_to_export)
                logger.debug(
                    "_do_work on Thread %s: Dispatched span %s after its parent %s with FlowContext %s",
                    threading.get_ident(),
                    span_to_export.attributes,
                    trace_metadata["trace_parent_id"],
                    trace_metadata,
                )
            else:
                # Requeue the Span to be uploaded later
                self._upload_queue.put((span_to_export, evaluation_context))
            self._upload_queue.task_done()

    def _export_span_dispatch(self, span: ReadableSpan) -> None:
        hl_file = read_from_opentelemetry_span(span, key=HUMANLOOP_FILE_KEY)
        file_type = span._attributes.get(HUMANLOOP_FILE_TYPE_KEY)  # type: ignore

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
        trace_metadata = TRACE_FLOW_CONTEXT.get(span.get_span_context().span_id)
        if trace_metadata and "trace_parent_id" in trace_metadata and trace_metadata["trace_parent_id"]:
            trace_parent_id = self._span_id_to_uploaded_log_id[trace_metadata["trace_parent_id"]]
            if trace_parent_id is None:
                # Parent Log in Trace upload failed
                file_path = read_from_opentelemetry_span(span, key=HUMANLOOP_PATH_KEY)
                logger.error(f"Skipping log for {file_path}: parent Log upload failed")
                return
        else:
            trace_parent_id = None
        prompt: PromptKernelRequestParams = file_object["prompt"]
        path: str = file_object["path"]
        if "output" in log_object:
            if not isinstance(log_object["output"], str):
                # Output expected to be a string, if decorated function
                # does not return one, jsonify it
                log_object["output"] = json.dumps(log_object["output"])
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

    def _export_tool(self, span: ReadableSpan) -> None:
        file_object: dict[str, Any] = read_from_opentelemetry_span(span, key=HUMANLOOP_FILE_KEY)
        log_object: dict[str, Any] = read_from_opentelemetry_span(span, key=HUMANLOOP_LOG_KEY)
        trace_metadata: FlowContext = TRACE_FLOW_CONTEXT.get(span.get_span_context().span_id, {})
        if "trace_parent_id" in trace_metadata and trace_metadata["trace_parent_id"]:
            trace_parent_id = self._span_id_to_uploaded_log_id.get(
                trace_metadata["trace_parent_id"],
            )
            if trace_parent_id is None:
                # Parent Log in Trace upload failed
                file_path = read_from_opentelemetry_span(span, key=HUMANLOOP_PATH_KEY)
                logger.error(f"Skipping log for {file_path}: parent Log upload failed")
                return
        else:
            trace_parent_id = None
        tool = file_object["tool"]
        if not tool.get("attributes"):
            tool["attributes"] = {}
        if not tool.get("setup_values"):
            tool["setup_values"] = {}
        path: str = file_object["path"]
        if "parameters" in tool["function"] and "properties" not in tool["function"]["parameters"]:
            tool["function"]["parameters"]["properties"] = {}
        if not isinstance(log_object["output"], str):
            # Output expected to be a string, if decorated function
            # does not return one, jsonify it
            log_object["output"] = json.dumps(log_object["output"])
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

    def _export_flow(self, span: ReadableSpan) -> None:
        file_object: dict[str, Any] = read_from_opentelemetry_span(span, key=HUMANLOOP_FILE_KEY)
        log_object: dict[str, Any] = read_from_opentelemetry_span(span, key=HUMANLOOP_LOG_KEY)
        trace_metadata: FlowContext = TRACE_FLOW_CONTEXT.get(
            span.get_span_context().span_id,
            {},
        )
        if "trace_parent_id" in trace_metadata:
            trace_parent_id = self._span_id_to_uploaded_log_id.get(
                trace_metadata["trace_parent_id"],  # type: ignore
            )
            if trace_parent_id is None and trace_metadata["trace_id"] != span.get_span_context().span_id:
                # Parent Log in Trace upload failed
                # NOTE: Check if the trace_id metadata field points to the
                # span itself. This signifies the span is the head of the Trace
                file_path = read_from_opentelemetry_span(span, key=HUMANLOOP_PATH_KEY)
                logger.error(f"Skipping log for {file_path}: parent Log upload failed")
                return
        else:
            trace_parent_id = None
        flow: FlowKernelRequestParams
        if not file_object.get("flow"):
            flow = {"attributes": {}}
        else:
            flow = file_object["flow"]
        path: str = file_object["path"]
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
