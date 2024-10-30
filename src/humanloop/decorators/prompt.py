import uuid
from functools import wraps
from typing import Any, Callable, Optional

from humanloop.otel import get_trace_context, get_tracer, pop_trace_context, push_trace_context
from humanloop.otel.constants import HL_FILE_OT_KEY, HL_LOG_OT_KEY, HL_TRACE_METADATA_KEY
from humanloop.otel.helpers import write_to_opentelemetry_span
from humanloop.types.model_endpoints import ModelEndpoints
from humanloop.types.model_providers import ModelProviders
from humanloop.types.prompt_kernel_request_stop import PromptKernelRequestStop
from humanloop.types.prompt_kernel_request_template import PromptKernelRequestTemplate
from humanloop.types.response_format import ResponseFormat


def prompt(
    path: Optional[str] = None,
    # TODO: Template can be a list of objects?
    model: Optional[str] = None,
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
):
    def decorator(func: Callable):
        if temperature is not None:
            if not 0 <= temperature < 1:
                raise ValueError(f"{func.__name__}: Temperature parameter must be between 0 and 1")

        if top_p is not None:
            if not 0 <= top_p <= 1:
                raise ValueError(f"{func.__name__}: Top-p parameter must be between 0 and 1")

        if presence_penalty is not None:
            if not -2 <= presence_penalty <= 2:
                raise ValueError(f"{func.__name__}: Presence penalty parameter must be between -2 and 2")

        if frequency_penalty is not None:
            if not -2 <= frequency_penalty <= 2:
                raise ValueError(f"{func.__name__}: Frequency penalty parameter must be between -2 and 2")

        @wraps(func)
        def wrapper(*args, **kwargs):
            with get_tracer().start_as_current_span(str(uuid.uuid4())) as span:
                trace_metadata = get_trace_context()

                if trace_metadata:
                    # We are in a Flow context
                    write_to_opentelemetry_span(
                        span=span,
                        key=HL_TRACE_METADATA_KEY,
                        value={**trace_metadata, "is_flow_log": False},
                    )
                    # Set current Prompt to act as parent for Logs nested underneath
                    push_trace_context(
                        {
                            **trace_metadata,
                            "trace_parent_id": span.get_span_context().span_id,
                            "is_flow_log": False,
                        },
                    )

                write_to_opentelemetry_span(
                    span=span,
                    key=HL_FILE_OT_KEY,
                    value={
                        "path": path if path else func.__name__,
                        "prompt": {
                            "template": template,
                            "temperature": temperature,
                            "top_p": top_p,
                            "presence_penalty": presence_penalty,
                            "frequency_penalty": frequency_penalty,
                            "model": model,
                            "endpoint": endpoint,
                            "provider": provider,
                            "max_tokens": max_tokens,
                            "stop": stop,
                            "other": other,
                            "seed": seed,
                            "response_format": response_format,
                        },
                    },
                )

                try:
                    output = func(*args, **kwargs)
                except Exception as e:
                    # TODO Some fails coming from here, they result in a fast end or duplicate
                    # spans outputted to the Humanloop API
                    print(e)
                    span.record_exception(e)
                    output = None

                if trace_metadata:
                    # Go back to previous trace context in Trace context
                    pop_trace_context()

                hl_log = {}
                if output:
                    hl_log["output"] = output
                write_to_opentelemetry_span(
                    span=span,
                    key=HL_LOG_OT_KEY,
                    value=hl_log,
                )

            return output

        return wrapper

    return decorator
