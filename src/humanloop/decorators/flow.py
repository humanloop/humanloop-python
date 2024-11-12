import logging
from functools import wraps
from typing import Any, Callable, Mapping, Optional, Sequence

from opentelemetry.sdk.trace import Span
from opentelemetry.trace import Tracer
from typing_extensions import Unpack

from humanloop.decorators.helpers import args_to_inputs
from humanloop.eval_utils.types import File
from humanloop.otel import TRACE_FLOW_CONTEXT, FlowContext
from humanloop.otel.constants import HUMANLOOP_FILE_KEY, HUMANLOOP_FILE_TYPE_KEY, HUMANLOOP_LOG_KEY, HUMANLOOP_PATH_KEY
from humanloop.otel.helpers import generate_span_id, jsonify_if_not_string, write_to_opentelemetry_span
from humanloop.requests import FlowKernelRequestParams as FlowDict
from humanloop.requests.flow_kernel_request import FlowKernelRequestParams

logger = logging.getLogger("humanloop.sdk")


def flow(
    opentelemetry_tracer: Tracer,
    path: Optional[str] = None,
    **flow_kernel: Unpack[FlowKernelRequestParams],  # type: ignore
):
    flow_kernel["attributes"] = {k: v for k, v in flow_kernel.get("attributes", {}).items() if v is not None}

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args: Sequence[Any], **kwargs: Mapping[str, Any]) -> Any:
            span: Span
            with opentelemetry_tracer.start_as_current_span(generate_span_id()) as span:  # type: ignore
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

                span.set_attribute(HUMANLOOP_PATH_KEY, path if path else func.__name__)
                span.set_attribute(HUMANLOOP_FILE_TYPE_KEY, "flow")
                if flow_kernel:
                    write_to_opentelemetry_span(
                        span=span,
                        key=f"{HUMANLOOP_FILE_KEY}.flow",
                        value=flow_kernel,  # type: ignore
                    )

                inputs = args_to_inputs(func, args, kwargs)

                # Call the decorated function
                try:
                    output = func(*args, **kwargs)
                    output_stringified = jsonify_if_not_string(
                        func=func,
                        output=output,
                    )
                    error = None
                except Exception as e:
                    logger.error(f"Error calling {func.__name__}: {e}")
                    output = None
                    output_stringified = jsonify_if_not_string(
                        func=func,
                        output=None,
                    )
                    error = str(e)

                flow_log = {
                    "inputs": inputs,
                    "output": output_stringified,
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
                        key=HUMANLOOP_LOG_KEY,
                        value=flow_log,  # type: ignore
                    )

            # Return the output of the decorated function
            return output

        wrapper.file = File(  # type: ignore
            path=path if path else func.__name__,
            type="flow",
            version=FlowDict(**flow_kernel),  # type: ignore
            callable=wrapper,
        )

        return wrapper

    return decorator
