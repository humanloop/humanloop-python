import logging
import os
import sys
import typing
import uuid
from functools import wraps
from typing import Any, Callable, Mapping, Optional, Sequence

from opentelemetry.sdk.trace import Span
from opentelemetry.trace import Tracer

from humanloop.types.prompt_kernel_request import PromptKernelRequest

if typing.TYPE_CHECKING:
    from humanloop import ToolFunctionParams
from humanloop.decorators.helpers import args_to_inputs
from humanloop.eval_utils import File
from humanloop.otel import TRACE_FLOW_CONTEXT, FlowContext
from humanloop.otel.constants import HUMANLOOP_FILE_KEY, HUMANLOOP_FILE_TYPE_KEY, HUMANLOOP_LOG_KEY, HUMANLOOP_PATH_KEY
from humanloop.otel.helpers import generate_span_id, write_to_opentelemetry_span
from humanloop.types.model_endpoints import ModelEndpoints
from humanloop.types.model_providers import ModelProviders
from humanloop.types.prompt_kernel_request_stop import PromptKernelRequestStop
from humanloop.types.prompt_kernel_request_template import PromptKernelRequestTemplate
from humanloop.types.response_format import ResponseFormat

logger = logging.getLogger("humanloop.sdk")


def prompt(
    opentelemetry_tracer: Tracer,
    path: Optional[str] = None,
    # TODO: Template can be a list of objects?
    model: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
    endpoint: Optional[ModelEndpoints] = None,
    template: Optional[PromptKernelRequestTemplate] = None,
    provider: Optional[ModelProviders] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[PromptKernelRequestStop] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    other: Optional[dict[str, Optional[Any]]] = None,
    seed: Optional[int] = None,
    response_format: Optional[ResponseFormat] = None,
    tools: Optional[Sequence["ToolFunctionParams"]] = None,
):
    def decorator(func: Callable):
        prompt_kernel = {}
        for attr_name, attr_value in {
            "model": model,
            "endpoint": endpoint,
            "template": template,
            "provider": provider,
            "max_tokens": max_tokens,
            "stop": stop,
            "other": other,
            "seed": seed,
            "response_format": response_format,
            "attributes": attributes or None,
            "tools": tools,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }.items():
            prompt_kernel[attr_name] = attr_value  # type: ignore

        @wraps(func)
        def wrapper(*args: Sequence[Any], **kwargs: Mapping[str, Any]) -> Any:
            span: Span
            with opentelemetry_tracer.start_as_current_span(generate_span_id()) as span:
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
                        value=prompt_kernel,  # type: ignore
                    )

                # Call the decorated function
                try:
                    output = func(*args, **kwargs)
                    error = None
                except Exception as e:
                    logger.error(f"Error calling {func.__name__}: {e}")
                    output = None
                    error = str(e)

                prompt_log = {
                    "inputs": args_to_inputs(func, args, kwargs),
                    "output": output,
                    "error": error,
                }
                write_to_opentelemetry_span(
                    span=span,
                    key=HUMANLOOP_LOG_KEY,
                    value=prompt_log,
                )

            # Return the output of the decorated function
            return output

        prompt_kernel_file = {**prompt_kernel}
        if prompt_kernel_file.get("provider") is None:
            prompt_kernel_file["provider"] = "openai"  # type: ignore
        if prompt_kernel_file.get("endpoint") is None:
            prompt_kernel_file["endpoint"] = "chat"  # type: ignore

        wrapper.file = File(  # type: ignore
            path=path if path else func.__name__,
            type="prompt",
            version={**prompt_kernel_file},  # type: ignore
            callable=wrapper,
        )

        return wrapper

    return decorator
