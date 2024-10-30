import uuid
from functools import wraps
from typing import Any, Callable, Optional

from humanloop.otel import get_humanloop_sdk_tracer, get_trace_parent_metadata, pop_trace_context, push_trace_context
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
                        value={**trace_metadata, "is_flow_log": False},
                    )
                    # Add Trace metadata to the context for the children
                    # Spans to be able to link to the parent Span
                    push_trace_context(
                        {
                            **trace_metadata,
                            "trace_parent_id": span.get_span_context().span_id,
                            "is_flow_log": False,
                        },
                    )

                # Write the Prompt Kernel to the Span on HL_FILE_OT_KEY
                write_to_opentelemetry_span(
                    span=span,
                    key=HL_FILE_OT_KEY,
                    value={
                        "path": path if path else func.__name__,
                        # Values not specified in the decorator will be
                        # completed with the intercepted values from the
                        # Instrumentors for LLM providers
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

                # Call the decorated function
                output = func(*args, **kwargs)

                # All children Spans have been created when the decorated function returns
                # Remove the Trace metadata from the context so the siblings can have
                # their children linked properly
                if trace_metadata:
                    # Go back to previous trace context in Trace context
                    pop_trace_context()

                prompt_log = {}
                if output:
                    prompt_log["output"] = output

                # Write the Prompt Log to the Span on HL_LOG_OT_KEY
                write_to_opentelemetry_span(
                    span=span,
                    key=HL_LOG_OT_KEY,
                    value=prompt_log,
                )

            # Return the output of the decorated function
            return output

        return wrapper

    return decorator
