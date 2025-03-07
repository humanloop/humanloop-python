import logging

import time
import typing
from queue import Empty as EmptyQueue
from queue import Queue
from threading import Thread
from typing import Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

import requests
from typing import Callable
from humanloop.context import get_evaluation_context
from humanloop.evals.run import HumanloopRuntimeError
from humanloop.otel.constants import (
    HUMANLOOP_FILE_TYPE_KEY,
    HUMANLOOP_LOG_KEY,
    HUMANLOOP_FILE_PATH_KEY,
)
from humanloop.otel.exporter.proto import serialize_span
from humanloop.otel.helpers import (
    read_from_opentelemetry_span,
    write_to_opentelemetry_span,
)


if typing.TYPE_CHECKING:
    from humanloop.client import Humanloop


logger = logging.getLogger("humanloop.sdk")


class HumanloopSpanExporter(SpanExporter):
    DEFAULT_NUMBER_THREADS = 1

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
            file_type = span.attributes.get(HUMANLOOP_FILE_TYPE_KEY)  # type: ignore [union-attr]
            if file_type is None:
                raise HumanloopRuntimeError("Internal error: Span does not have type set")

            try:
                log_args = read_from_opentelemetry_span(
                    span=span,
                    key=HUMANLOOP_LOG_KEY,
                )
                path = read_from_opentelemetry_span(
                    span=span,
                    key=HUMANLOOP_FILE_PATH_KEY,
                )
                evaluation_context = get_evaluation_context()
                if evaluation_context is not None:
                    kwargs_eval, eval_callback = evaluation_context.log_args_with_context(
                        path=path,  # type: ignore [arg-type]
                        log_args=log_args,  # type: ignore [arg-type]
                    )
                    write_to_opentelemetry_span(
                        span=span,
                        key=HUMANLOOP_LOG_KEY,
                        value=kwargs_eval,
                    )
                else:
                    eval_callback = None
            except HumanloopRuntimeError as e:
                raise e
            except Exception:
                # No log args in the span
                eval_callback = None

            self._upload_queue.put((span, eval_callback))

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
        # Do work while the Exporter was not instructed to
        # wind down or the queue is not empty
        while self._upload_queue.qsize() > 0 or not self._shutdown:
            thread_args: tuple[ReadableSpan, Optional[Callable[[str], None]]]  # type: ignore
            try:
                # Don't block or the thread will never be notified of the shutdown
                thread_args = self._upload_queue.get(block=False)  # type: ignore
            except EmptyQueue:
                # Wait for another span to arrive
                time.sleep(0.1)
                continue

            span_to_export, eval_context_callback = thread_args

            response = requests.post(
                f"{self._client._client_wrapper.get_base_url()}/import/otel/v1/traces",
                headers={
                    **self._client._client_wrapper.get_headers(),
                },
                data=serialize_span(span_to_export),
            )
            if response.status_code != 200:
                raise HumanloopRuntimeError(
                    f"Failed to upload OTEL span to Humanloop: {response.json()} {response.status_code}"
                )
            else:
                if eval_context_callback:
                    log_id = response.json()["records"][0]
                    eval_context_callback(log_id)

            self._upload_queue.task_done()
