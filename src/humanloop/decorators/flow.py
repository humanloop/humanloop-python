import uuid
from functools import wraps
from typing import Any, Callable, Optional

from humanloop.decorators.helpers import args_to_inputs
from humanloop.otel import get_trace_context, get_tracer, pop_trace_context, push_trace_context
from humanloop.otel.constants import HL_FILE_OT_KEY, HL_LOG_OT_KEY, HL_TRACE_METADATA_KEY, OT_EMPTY_ATTRIBUTE
from humanloop.otel.helpers import write_to_opentelemetry_span


def flow(
    path: Optional[str] = None,
    attributes: dict[str, Any] = {},
):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with get_tracer().start_as_current_span(str(uuid.uuid4())) as span:
                trace_metadata = get_trace_context()

                if trace_metadata:
                    write_to_opentelemetry_span(
                        span=span,
                        key=HL_TRACE_METADATA_KEY,
                        value={
                            **trace_metadata,
                            "trace_id": span.get_span_context().span_id,
                            "is_flow_log": True,
                        },
                    )
                else:
                    write_to_opentelemetry_span(
                        span=span,
                        key=HL_TRACE_METADATA_KEY,
                        value={
                            "trace_id": span.get_span_context().span_id,
                            "is_flow_log": True,
                        },
                    )
                # Set this as the Flow to which Logs are appended
                # Important: Flows might be nested under each other
                push_trace_context(
                    {
                        "trace_id": span.get_span_context().span_id,
                        "trace_parent_id": span.get_span_context().span_id,
                        "is_flow_log": True,
                    },
                )

                result = func(*args, **kwargs)

                pop_trace_context()

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
                write_to_opentelemetry_span(
                    span=span,
                    key=HL_LOG_OT_KEY,
                    value={
                        "inputs": args_to_inputs(func, args, kwargs),
                        "output": result,
                    },
                )

            return result

        return wrapper

    return decorator
