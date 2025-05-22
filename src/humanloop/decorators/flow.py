import logging
import typing
from functools import wraps
from typing import Any, Awaitable, Callable, Literal, Optional, TypeVar, Union, overload

from opentelemetry.trace import Span, Tracer
from typing_extensions import ParamSpec

from humanloop.base_client import BaseHumanloop
from humanloop.context import (
    DecoratorContext,
    get_trace_id,
    set_decorator_context,
    set_trace_id,
)
from humanloop.decorators.helpers import bind_args
from humanloop.evals.run import HumanloopRuntimeError
from humanloop.evals.types import FileEvalConfig
from humanloop.otel.constants import (
    HUMANLOOP_FILE_PATH_KEY,
    HUMANLOOP_FILE_TYPE_KEY,
    HUMANLOOP_FLOW_SPAN_NAME,
    HUMANLOOP_LOG_KEY,
)
from humanloop.otel.helpers import process_output, write_to_opentelemetry_span
from humanloop.requests import FlowKernelRequestParams as FlowDict
from humanloop.types.chat_message import ChatMessage
from humanloop.types.flow_log_response import FlowLogResponse

logger = logging.getLogger("humanloop.sdk")


P = ParamSpec("P")
R = TypeVar("R")


def flow_decorator_factory(
    client: "BaseHumanloop",
    opentelemetry_tracer: Tracer,
    path: str,
    attributes: Optional[dict[str, Any]] = None,
):
    def decorator(func: Callable[P, R]) -> Callable[P, Optional[R]]:
        decorator_path = path or func.__name__
        file_type = "flow"
        flow_kernel = {"attributes": attributes or {}}

        wrapper = _wrapper_factory(
            client=client,
            opentelemetry_tracer=opentelemetry_tracer,
            func=func,
            path=path,
            flow_kernel=flow_kernel,
            is_awaitable=False,
        )

        wrapper.file = FileEvalConfig(  # type: ignore
            path=decorator_path,
            type=file_type,  # type: ignore [arg-type, typeddict-item]
            version=FlowDict(**flow_kernel),  # type: ignore
            callable=wrapper,
        )

        return wrapper

    return decorator


def a_flow_decorator_factory(
    client: "BaseHumanloop",
    opentelemetry_tracer: Tracer,
    path: str,
    attributes: Optional[dict[str, Any]] = None,
):
    def decorator(func: Callable[P, Awaitable[R]]):
        decorator_path = path or func.__name__
        file_type = "flow"
        flow_kernel = {"attributes": attributes or {}}

        wrapper = _wrapper_factory(
            client=client,
            opentelemetry_tracer=opentelemetry_tracer,
            func=func,
            path=path,
            flow_kernel=flow_kernel,
            is_awaitable=True,
        )

        wrapper.file = FileEvalConfig(  # type: ignore
            path=decorator_path,
            type=file_type,  # type: ignore [arg-type, typeddict-item]
            version=FlowDict(**flow_kernel),  # type: ignore
            callable=wrapper,
        )

        return wrapper

    return decorator


@overload
def _wrapper_factory(
    client: "BaseHumanloop",
    opentelemetry_tracer: Tracer,
    func: Callable[P, Awaitable[R]],
    path: str,
    flow_kernel: dict[str, Any],
    is_awaitable: Literal[True],
) -> Callable[P, Awaitable[Optional[R]]]: ...


@overload
def _wrapper_factory(
    client: "BaseHumanloop",
    opentelemetry_tracer: Tracer,
    func: Callable[P, R],
    path: str,
    flow_kernel: dict[str, Any],
    is_awaitable: Literal[False],
) -> Callable[P, Optional[R]]: ...


def _wrapper_factory(  # type: ignore [misc]
    client: "BaseHumanloop",
    opentelemetry_tracer: Tracer,
    func: Union[Callable[P, Awaitable[R]], Callable[P, R]],
    path: str,
    flow_kernel: dict[str, Any],
    is_awaitable: bool,
):
    if is_awaitable:
        func = typing.cast(Callable[P, Awaitable[R]], func)

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Optional[R]:
            span: Span
            with set_decorator_context(
                DecoratorContext(
                    path=path,
                    type="flow",
                    version=flow_kernel,
                )
            ) as decorator_context:
                with opentelemetry_tracer.start_as_current_span(HUMANLOOP_FLOW_SPAN_NAME) as span:  # type: ignore
                    span, flow_log = _process_inputs(
                        client=client,
                        span=span,
                        decorator_context=decorator_context,
                        decorator_path=path,
                        file_type="flow",
                        func=func,
                        args=args,
                        kwargs=kwargs,
                    )

                    with set_trace_id(flow_log.id):
                        func_output: Optional[R]
                        try:
                            func_output = await func(*args, **kwargs)  # type: ignore [misc]
                            error = None
                        except HumanloopRuntimeError as e:
                            # Critical error, re-raise
                            client.logs.delete(id=flow_log.id)
                            span.record_exception(e)
                            raise e
                        except Exception as e:
                            logger.error(f"Error calling {func.__name__}: {e}")
                            error = e
                            func_output = None

                        _process_output(
                            func=func,
                            span=span,
                            func_output=func_output,
                            error=error,
                            flow_log=flow_log,
                        )

                        # Return the output of the decorated function
                        return func_output
    else:
        func = typing.cast(Callable[P, R], func)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Optional[R]:
            span: Span
            with set_decorator_context(
                DecoratorContext(
                    path=path,
                    type="flow",
                    version=flow_kernel,
                )
            ) as decorator_context:
                with opentelemetry_tracer.start_as_current_span(HUMANLOOP_FLOW_SPAN_NAME) as span:  # type: ignore
                    span, flow_log = _process_inputs(
                        client=client,
                        span=span,
                        decorator_context=decorator_context,
                        decorator_path=path,
                        file_type="flow",
                        func=func,
                        args=args,
                        kwargs=kwargs,
                    )

                    with set_trace_id(flow_log.id):
                        func_output: Optional[R]
                        try:
                            func_output = func(*args, **kwargs)
                            error = None
                        except HumanloopRuntimeError as e:
                            # Critical error, re-raise
                            client.logs.delete(id=flow_log.id)
                            span.record_exception(e)
                            raise e
                        except Exception as e:
                            logger.error(f"Error calling {func.__name__}: {e}")
                            error = e
                            func_output = None

                        _process_output(
                            func=func,
                            span=span,
                            func_output=func_output,
                            error=error,
                            flow_log=flow_log,
                        )

                        # Return the output of the decorated function
                        return func_output

    return wrapper


def _process_inputs(
    client: "BaseHumanloop",
    span: Span,
    decorator_context: DecoratorContext,
    decorator_path: str,
    file_type: str,
    func: Callable[P, R],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
):
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
    flow_log: FlowLogResponse = client.flows._log(  # type: ignore [attr-defined]
        path=decorator_context.path,
        flow=decorator_context.version,
        log_status="incomplete",
        **init_log_inputs,
    )
    return span, flow_log


def _process_output(
    func: Union[Callable[P, R], Callable[P, Awaitable[R]]],
    span: Span,
    func_output: Optional[R],
    error: Optional[Exception],
    flow_log: FlowLogResponse,
):
    log_output: Optional[str]
    log_error: Optional[str]
    log_output_message: Optional[ChatMessage]
    if not error:
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
    else:
        log_output = None
        log_output_message = None
        log_error = str(error)
        func_output = None
    updated_flow_log = {
        "log_status": "complete",
        "output": log_output,
        "error": log_error,
        "output_message": log_output_message,
        "id": flow_log.id,
    }
    # Write the Flow Log to the Span on HL_LOG_OT_KEY
    write_to_opentelemetry_span(
        span=span,  # type: ignore [arg-type]
        key=HUMANLOOP_LOG_KEY,
        value=updated_flow_log,  # type: ignore
    )
