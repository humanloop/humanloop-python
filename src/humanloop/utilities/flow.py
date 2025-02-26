import logging
from functools import wraps
from typing import Any, Callable, Mapping, Optional, Sequence, TypeVar
from typing_extensions import ParamSpec

from opentelemetry.trace import Span, Tracer
from opentelemetry import context as context_api
import requests

from humanloop.base_client import BaseHumanloop
from humanloop.context import get_trace_id, set_trace_id
from humanloop.types.chat_message import ChatMessage
from humanloop.utilities.helpers import bind_args
from humanloop.eval_utils.types import File
from humanloop.otel.constants import (
    HUMANLOOP_FILE_TYPE_KEY,
    HUMANLOOP_LOG_KEY,
    HUMANLOOP_PATH_KEY,
)
from humanloop.otel.helpers import process_output, write_to_opentelemetry_span
from humanloop.requests import FlowKernelRequestParams as FlowDict

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

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        decorator_path = path or func.__name__
        file_type = "flow"

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Optional[R]:
            span: Span
            with opentelemetry_tracer.start_as_current_span("humanloop.flow") as span:  # type: ignore
                trace_id = get_trace_id()
                args_to_func = bind_args(func, args, kwargs)

                # Create the trace ahead so we have a parent ID to reference
                init_log_inputs = {
                    "inputs": {k: v for k, v in args_to_func.items() if k != "messages"},
                    "messages": args_to_func.get("messages"),
                    "trace_parent_id": trace_id,
                }
                init_log = requests.post(
                    f"{client._client_wrapper.get_base_url()}/flows/log",
                    headers=client._client_wrapper.get_headers(),
                    json={
                        "path": path,
                        "flow": flow_kernel,
                        "log_status": "incomplete",
                        **init_log_inputs,
                    },
                ).json()
                # log = client.flows.log(
                #     path=path,
                #     **log_inputs,
                #     log_status="incomplete",
                # )
                token = set_trace_id(init_log["id"])

                span.set_attribute(HUMANLOOP_PATH_KEY, decorator_path)
                span.set_attribute(HUMANLOOP_FILE_TYPE_KEY, file_type)

                func_output: Optional[R]
                log_output: str
                log_error: Optional[str]
                log_output_message: ChatMessage
                try:
                    func_output = func(*args, **kwargs)
                    if (
                        isinstance(func_output, dict)
                        and len(func_output.keys()) == 2
                        and "role" in func_output
                        and "content" in func_output
                    ):
                        log_output_message = ChatMessage(**func_output)
                        log_output = None
                    else:
                        log_output = process_output(func=func, output=func_output)
                        log_output_message = None
                    log_error = None
                except Exception as e:
                    logger.error(f"Error calling {func.__name__}: {e}")
                    output = None
                    log_output_message = None
                    log_error = str(e)

                flow_log = {
                    "inputs": {k: v for k, v in args_to_func.items() if k != "messages"},
                    "messages": args_to_func.get("messages"),
                    "log_status": "complete",
                    "output": log_output,
                    "error": log_error,
                    "output_message": log_output_message,
                    "id": init_log["id"],
                }

                # Write the Flow Log to the Span on HL_LOG_OT_KEY
                if flow_log:
                    write_to_opentelemetry_span(
                        span=span,
                        key=HUMANLOOP_LOG_KEY,
                        value=flow_log,  # type: ignore
                    )

            context_api.detach(token=token)
            # Return the output of the decorated function
            return output

        wrapper.file = File(  # type: ignore
            path=decorator_path,
            type=file_type,
            version=FlowDict(**flow_kernel),  # type: ignore
            callable=wrapper,
        )

        return wrapper

    return decorator
