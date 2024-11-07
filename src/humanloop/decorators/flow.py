import logging
import uuid
from functools import wraps
from typing import Any, Callable, Mapping, Optional, Sequence

from opentelemetry.sdk.trace import Span
from opentelemetry.trace import Tracer
from opentelemetry.util.types import AttributeValue

from humanloop.decorators.helpers import args_to_inputs
from humanloop.eval_utils.types import File
from humanloop.otel import TRACE_FLOW_CONTEXT, FlowContext
from humanloop.otel.constants import HL_FILE_KEY, HL_FILE_TYPE_KEY, HL_LOG_KEY, HL_PATH_KEY
from humanloop.otel.helpers import write_to_opentelemetry_span
from humanloop.requests import FlowKernelRequestParams as FlowDict

logger = logging.getLogger("humanloop.sdk")


def flow(
    opentelemetry_tracer: Tracer,
    path: Optional[str] = None,
    attributes: Optional[dict[str, AttributeValue]] = None,
):
    if attributes is None:
        attributes = {}
    attributes = {k: v for k, v in attributes.items() if v is not None}

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args: Sequence[Any], **kwargs: Mapping[str, Any]) -> Any:
            span: Span
            with opentelemetry_tracer.start_as_current_span(str(uuid.uuid4())) as span:
                span_id = span.get_span_context().span_id
                if span.parent:
                    span_parent_id = span.parent.span_id
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

                span.set_attribute(HL_PATH_KEY, path if path else func.__name__)
                span.set_attribute(HL_FILE_TYPE_KEY, "flow")
                if attributes:
                    write_to_opentelemetry_span(
                        span=span,
                        key=f"{HL_FILE_KEY}.flow.attributes",
                        value=attributes,  # type: ignore
                    )

                inputs = args_to_inputs(func, args, kwargs)

                # Call the decorated function
                try:
                    output = func(*args, **kwargs)
                    error = None
                except Exception as e:
                    logger.error(f"{func.__name__}: {e}")
                    output = None
                    error = str(e)

                flow_log = {
                    "inputs": inputs,
                    "output": output,
                    "error": error,
                }
                if inputs:
                    flow_log["inputs"] = inputs
                if output:
                    flow_log["output"] = output

                # Write the Flow Log to the Span on HL_LOG_OT_KEY
                if flow_log:
                    write_to_opentelemetry_span(
                        span=span,
                        key=HL_LOG_KEY,
                        value=flow_log,  # type: ignore
                    )

            # Return the output of the decorated function
            return output

        func.file = File(  # type: ignore
            path=path if path else func.__name__,
            type="flow",
            version=FlowDict(attributes=attributes),  # type: ignore
            is_decorated=True,
            callable=wrapper,
        )

        return wrapper

    return decorator
