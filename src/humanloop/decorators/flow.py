import uuid
from functools import wraps
from typing import Any, Callable, Optional

from humanloop.decorators.helpers import args_to_inputs
from humanloop.otel import get_humanloop_sdk_tracer, get_trace_parent_metadata, pop_trace_context, push_trace_context
from humanloop.otel.constants import HL_FILE_OT_KEY, HL_LOG_OT_KEY, HL_TRACE_METADATA_KEY, OT_EMPTY_ATTRIBUTE
from humanloop.otel.helpers import write_to_opentelemetry_span


def flow(
    path: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
):
    if attributes is None:
        attributes = {}

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with get_humanloop_sdk_tracer().start_as_current_span(str(uuid.uuid4())) as span:
                trace_metadata = get_trace_parent_metadata()

                if trace_metadata:
                    # Add Trace metadata to the Span so it can be correctly
                    # linked to the parent Span. trace_metadata will be
                    # non-null if the function is called by a @flow
                    # decorated function.
                    write_to_opentelemetry_span(
                        span=span,
                        key=HL_TRACE_METADATA_KEY,
                        value={
                            "trace_parent_id": trace_metadata["trace_parent_id"],
                            "trace_id": span.get_span_context().span_id,
                            "is_flow_log": True,
                        },
                    )
                else:
                    # The Flow Log is not nested under another Flow Log
                    # Set the trace_id to the current span_id
                    write_to_opentelemetry_span(
                        span=span,
                        key=HL_TRACE_METADATA_KEY,
                        value={
                            "trace_id": span.get_span_context().span_id,
                            "is_flow_log": True,
                        },
                    )

                # Add Trace metadata to the context for the children
                # Spans to be able to link to the parent Span
                # Unlike other decorators, which push to context stack
                # only if trace_metadata is present, this decorator
                # always pushes to context stack since it is responsible
                # for creating the context stack
                push_trace_context(
                    {
                        "trace_id": span.get_span_context().span_id,
                        "trace_parent_id": span.get_span_context().span_id,
                        "is_flow_log": True,
                    },
                )

                # Write the Flow Kernel to the Span on HL_FILE_OT_KEY
                write_to_opentelemetry_span(
                    span=span,
                    key=HL_FILE_OT_KEY,
                    value={
                        "path": path if path else func.__name__,
                        # OT span attributes are dropped if they are empty or null
                        # Add 'EMPTY' token value otherwise the 'flow' key will be dropped
                        "flow": {"attributes": attributes} if attributes else OT_EMPTY_ATTRIBUTE,
                    },
                )

                # Call the decorated function
                output = func(*args, **kwargs)

                # All children Spans have been created when the decorated function returns
                # Remove the Trace metadata from the context so the siblings can have
                # their children linked properly
                pop_trace_context()

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

        return wrapper

    return decorator
