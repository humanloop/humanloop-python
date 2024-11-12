import logging
from functools import wraps
from typing import Any, Callable, Mapping, Optional, Sequence

from opentelemetry.sdk.trace import Span
from opentelemetry.trace import Tracer
from typing_extensions import Unpack

from humanloop.decorators.helpers import args_to_inputs
from humanloop.decorators.types import DecoratorPromptKernelRequestParams
from humanloop.eval_utils import File
from humanloop.otel import TRACE_FLOW_CONTEXT, FlowContext
from humanloop.otel.constants import HUMANLOOP_FILE_KEY, HUMANLOOP_FILE_TYPE_KEY, HUMANLOOP_LOG_KEY, HUMANLOOP_PATH_KEY
from humanloop.otel.helpers import generate_span_id, jsonify_if_not_string, write_to_opentelemetry_span

logger = logging.getLogger("humanloop.sdk")


def prompt(
    opentelemetry_tracer: Tracer,
    path: Optional[str] = None,
    # TODO: Template can be a list of objects?
    **prompt_kernel: Unpack[DecoratorPromptKernelRequestParams],  # type: ignore
):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args: Sequence[Any], **kwargs: Mapping[str, Any]) -> Any:
            span: Span
            with opentelemetry_tracer.start_as_current_span(generate_span_id()) as span:  # type: ignore
                span_id = span.get_span_context().span_id
                if span.parent:
                    span_parent_id = span.parent.span_id
                    parent_trace_metadata = TRACE_FLOW_CONTEXT.get(span_parent_id, {})
                    if parent_trace_metadata:
                        TRACE_FLOW_CONTEXT[span_id] = FlowContext(
                            trace_id=parent_trace_metadata["trace_id"],
                            trace_parent_id=span_parent_id,
                            is_flow_log=False,
                        )

                span.set_attribute(HUMANLOOP_PATH_KEY, path if path else func.__name__)
                span.set_attribute(HUMANLOOP_FILE_TYPE_KEY, "prompt")

                if prompt_kernel:
                    write_to_opentelemetry_span(
                        span=span,
                        key=f"{HUMANLOOP_FILE_KEY}.prompt",
                        value={
                            **prompt_kernel,  # type: ignore
                            "attributes": prompt_kernel.get("attributes") or None,  # type: ignore
                        },  # type: ignore
                    )

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
                        output=output,
                    )
                    error = str(e)

                prompt_log = {
                    "inputs": args_to_inputs(func, args, kwargs),
                    "output": output_stringified,
                    "error": error,
                }
                write_to_opentelemetry_span(
                    span=span,
                    key=HUMANLOOP_LOG_KEY,
                    value=prompt_log,  # type: ignore
                )

            # Return the output of the decorated function
            return output

        wrapper.file = File(  # type: ignore
            path=path if path else func.__name__,
            type="prompt",
            version={**prompt_kernel},  # type: ignore
            callable=wrapper,
        )

        return wrapper

    return decorator
