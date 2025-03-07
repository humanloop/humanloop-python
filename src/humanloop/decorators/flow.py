import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar
from typing_extensions import ParamSpec

from opentelemetry.trace import Span, Tracer

from humanloop.base_client import BaseHumanloop
from humanloop.context import (
    DecoratorContext,
    get_trace_id,
    set_decorator_context,
    set_trace_id,
)
from humanloop.evals.run import HumanloopRuntimeError
from humanloop.types.chat_message import ChatMessage
from humanloop.decorators.helpers import bind_args
from humanloop.evals.types import File
from humanloop.otel.constants import (
    HUMANLOOP_FILE_TYPE_KEY,
    HUMANLOOP_LOG_KEY,
    HUMANLOOP_FILE_PATH_KEY,
    HUMANLOOP_FLOW_SPAN_NAME,
)
from humanloop.otel.helpers import process_output, write_to_opentelemetry_span
from humanloop.requests import FlowKernelRequestParams as FlowDict
from humanloop.types.flow_log_response import FlowLogResponse

logger = logging.getLogger("humanloop.sdk")


P = ParamSpec("P")
R = TypeVar("R")


def flow(
    client: "BaseHumanloop",
    opentelemetry_tracer: Tracer,
    path: str,
    attributes: Optional[dict[str, Any]] = None,
):
    flow_kernel = {"attributes": attributes or {}}

    def decorator(func: Callable[P, R]) -> Callable[P, Optional[R]]:
        decorator_path = path or func.__name__
        file_type = "flow"

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Optional[R]:
            span: Span
            with set_decorator_context(
                DecoratorContext(
                    path=decorator_path,
                    type="flow",
                    version=flow_kernel,
                )
            ) as decorator_context:
                with opentelemetry_tracer.start_as_current_span(HUMANLOOP_FLOW_SPAN_NAME) as span:  # type: ignore
                    span.set_attribute(HUMANLOOP_FILE_PATH_KEY, decorator_path)
                    span.set_attribute(HUMANLOOP_FILE_TYPE_KEY, file_type)
                    trace_id = get_trace_id()
                    func_args = bind_args(func, args, kwargs)

                    # Create the trace ahead so we have a parent ID to reference
                    init_log_inputs = {
                        "inputs": {k: v for k, v in func_args.items() if k != "messages"},
                        "messages": func_args.get("messages"),
                        "trace_parent_id": trace_id,
                    }
                    this_flow_log: FlowLogResponse = client.flows._log(  # type: ignore [attr-defined]
                        path=decorator_context.path,
                        flow=decorator_context.version,
                        log_status="incomplete",
                        **init_log_inputs,
                    )

                    with set_trace_id(this_flow_log.id):
                        func_output: Optional[R]
                        log_output: Optional[str]
                        log_error: Optional[str]
                        log_output_message: Optional[ChatMessage]
                        try:
                            func_output = func(*args, **kwargs)
                            if (
                                isinstance(func_output, dict)
                                and len(func_output.keys()) == 2
                                and "role" in func_output
                                and "content" in func_output
                            ):
                                log_output_message = func_output  # type: ignore [assignment]
                                log_output = None
                            else:
                                log_output = process_output(func=func, output=func_output)
                                log_output_message = None
                            log_error = None
                        except HumanloopRuntimeError as e:
                            # Critical error, re-raise
                            client.logs.delete(id=this_flow_log.id)
                            span.record_exception(e)
                            raise e
                        except Exception as e:
                            logger.error(f"Error calling {func.__name__}: {e}")
                            log_output = None
                            log_output_message = None
                            log_error = str(e)
                            func_output = None

                        updated_flow_log = {
                            "log_status": "complete",
                            "output": log_output,
                            "error": log_error,
                            "output_message": log_output_message,
                            "id": this_flow_log.id,
                        }
                        # Write the Flow Log to the Span on HL_LOG_OT_KEY
                        write_to_opentelemetry_span(
                            span=span,  # type: ignore [arg-type]
                            key=HUMANLOOP_LOG_KEY,
                            value=updated_flow_log,  # type: ignore
                        )
                        # Return the output of the decorated function
                        return func_output  # type: ignore [return-value]

        wrapper.file = File(  # type: ignore
            path=decorator_path,
            type=file_type,  # type: ignore [arg-type, typeddict-item]
            version=FlowDict(**flow_kernel),  # type: ignore
            callable=wrapper,
        )

        return wrapper

    return decorator
