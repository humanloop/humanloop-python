import uuid
from functools import wraps
from typing import Any, Callable, Optional

from opentelemetry.trace import Tracer
from opentelemetry.sdk.trace import ReadableSpan

from humanloop.decorators.helpers import args_to_inputs
from humanloop.eval_utils import File
from humanloop.otel import TRACE_FLOW_CONTEXT, FlowContext
from humanloop.otel.constants import (
    HL_FILE_OT_KEY,
    HL_LOG_OT_KEY,
    HL_OT_EMPTY_VALUE,
)
from humanloop.otel.helpers import write_to_opentelemetry_span


def flow(
    opentelemetry_tracer: Tracer,
    path: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
):
    if attributes is None:
        attributes = {}

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            span: ReadableSpan
            with opentelemetry_tracer.start_as_current_span(str(uuid.uuid4())) as span:
                span_id = span.get_span_context().span_id
                if span.parent:
                    span_parent_id = span.parent.span_id
                else:
                    span_parent_id = None
                parent_trace_metadata = TRACE_FLOW_CONTEXT.get(span_parent_id)
                if parent_trace_metadata:
                    TRACE_FLOW_CONTEXT[span_id] = FlowContext(
                        trace_id=span_id,
                        trace_parent_id=span_parent_id,
                        is_flow_log=True,
                    )
                else:
                    # The Flow Log is not nested under another Flow Log
                    # Set the trace_id to the current span_id
                    TRACE_FLOW_CONTEXT[span_id] = FlowContext(
                        trace_id=span_id,
                        trace_parent_id=None,
                        is_flow_log=True,
                    )

                # Write the Flow Kernel to the Span on HL_FILE_OT_KEY
                write_to_opentelemetry_span(
                    span=span,
                    key=HL_FILE_OT_KEY,
                    value={
                        "path": path if path else func.__name__,
                        # If a None write is attempted then the attribute is removed
                        # making it impossible to distinguish between a Flow Span and
                        # Spans not created by Humanloop (see humanloop.otel.helpers.is_humanloop_span)
                        "flow": {"attributes": attributes} if attributes else HL_OT_EMPTY_VALUE,
                    },
                )

                # Call the decorated function
                output = func(*args, **kwargs)

                # Write the Flow Log to the Span on HL_LOG_OT_KEY
                write_to_opentelemetry_span(
                    span=span,
                    key=HL_LOG_OT_KEY,
                    value={
                        "inputs": args_to_inputs(func, args, kwargs),
                        "output": output,
                    },
                )

            # Return the output of the decorated function
            return output

        func.file = File(  # type: ignore
            id=None,
            path=path if path else func.__name__,
            type="flow",
            version=attributes,
            is_decorated=True,
            callable=wrapper,
        )

        return wrapper

    return decorator
