import logging

import typing
from queue import Empty as EmptyQueue
from queue import Queue
from threading import Thread
from typing import Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

import requests
from humanloop.context import get_evaluation_context, EvaluationContext
from humanloop.otel.constants import (
    HUMANLOOP_FILE_TYPE_KEY,
    HUMANLOOP_LOG_KEY,
    HUMANLOOP_PATH_KEY,
)
from humanloop.otel.helpers import read_from_opentelemetry_span, write_to_opentelemetry_span
from opentelemetry.proto.common.v1.common_pb2 import KeyValue, Link
from opentelemetry.proto.trace.v1.trace_pb2 import (
    TracesData,
    ResourceSpans,
    InstrumentationScope,
    ScopeSpans,
    Span as ProtoBufferSpan,
)

if typing.TYPE_CHECKING:
    from humanloop.client import Humanloop


logger = logging.getLogger("humanloop.sdk")


class HumanloopSpanExporter(SpanExporter):
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

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._shutdown:
            logger.warning("[HumanloopSpanExporter] Shutting down, not accepting new spans")
            return SpanExportResult.FAILURE

        for span in spans:
            self._upload_queue.put((span, get_evaluation_context()))

        return SpanExportResult.SUCCESS

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
        # Do work while the Exporter was not instructed to
        # wind down or the queue is not empty
        while self._upload_queue.qsize() > 0 or not self._shutdown:
            thread_args: tuple[ReadableSpan, Optional[EvaluationContext]]  # type: ignore
            try:
                # Don't block or the thread will never be notified of the shutdown
                thread_args = self._upload_queue.get(block=False)  # type: ignore
            except EmptyQueue:
                # Wait for the another span to arrive
                continue

            span_to_export, evaluation_context = thread_args
            file_type = span_to_export.attributes.get(HUMANLOOP_FILE_TYPE_KEY)
            file_path = span_to_export.attributes.get(HUMANLOOP_PATH_KEY)
            if file_type is None:
                raise ValueError("Span does not have type set")

            log_args = read_from_opentelemetry_span(
                span=span_to_export,
                key=HUMANLOOP_LOG_KEY,
            )

            if evaluation_context:
                if file_path == evaluation_context.path:
                    log_args = {
                        **log_args,
                        "source_datapoint_id": evaluation_context.source_datapoint_id,
                        "run_id": evaluation_context.run_id,
                    }

            write_to_opentelemetry_span(
                span=span_to_export,
                key=HUMANLOOP_LOG_KEY,
                value=log_args,
            )

            payload = TracesData(
                resource_spans=[
                    ResourceSpans(
                        scope_spans=ScopeSpans(
                            scope=InstrumentationScope(
                                name="humanloop-otel",
                                version="0.1.0",
                            ),
                            spans=[
                                ProtoBufferSpan(
                                    trace_id=span_to_export.context.trace_id,
                                    span_id=span_to_export.span_id,
                                    name=span_to_export.name,
                                    kind=span_to_export.kind,
                                    start_time_unix_nano=span_to_export.start_time,
                                    end_time_unix_nano=span_to_export.end_time,
                                    attributes=[
                                        KeyValue(
                                            key=key,
                                            value=value,
                                        )
                                        for key, value in span_to_export.attributes.items()
                                    ],
                                    dropped_attributes_count=len(span_to_export.dropped_attributes),
                                    dropped_events_count=len(span_to_export.dropped_events),
                                    dropped_links_count=len(span_to_export.dropped_links),
                                    links=[
                                        Link(
                                            trace_id=link.trace_id,
                                            span_id=link.span_id,
                                            attributes=link.attributes,
                                        )
                                        for link in span_to_export.links
                                    ],
                                    events=[],
                                )
                            ],
                        )
                    )
                ]
            )

            response = requests.post(
                f"{self._client._client_wrapper.get_base_url()}/import/otel/v1/traces",
                headers=self._client._client_wrapper.get_headers(),
                data=span_to_export.to_json().encode("ascii"),
            )
            if response.status_code != 200:
                # TODO: handle
                pass
            else:
                print("FOO", response.json())
                if evaluation_context and file_path == evaluation_context.path:
                    log_id = response.json()["log_id"]
                    evaluation_context.callback(log_id)

            self._upload_queue.task_done()
