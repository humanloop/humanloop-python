import builtins
import inspect
import textwrap
import typing
import uuid
from functools import wraps
from typing import Any, Callable, Mapping, Optional, Sequence, TypedDict, Union

from humanloop.otel import get_trace_context, get_tracer, pop_trace_context, push_trace_context
from humanloop.otel.constants import HL_FILE_OT_KEY, HL_LOG_OT_KEY, HL_TRACE_METADATA_KEY
from humanloop.otel.helpers import write_to_opentelemetry_span
from humanloop.types.tool_function import ToolFunction
from humanloop.types.tool_kernel_request import ToolKernelRequest

from .helpers import args_to_inputs


def _extract_annotation_signature(annotation: typing.Type) -> Union[list, tuple]:
    origin = typing.get_origin(annotation)
    if origin is None:
        if annotation is inspect._empty:
            raise ValueError("Empty type hint annotation")
        return [annotation]
    if origin is list:
        inner_type = _extract_annotation_signature(typing.get_args(annotation)[0])
        return [origin, inner_type]
    if origin is dict:
        key_type = _extract_annotation_signature(typing.get_args(annotation)[0])
        value_type = _extract_annotation_signature(typing.get_args(annotation)[1])
        return [origin, key_type, value_type]
    if origin is tuple:
        return [
            origin,
            *[_extract_annotation_signature(arg) for arg in typing.get_args(annotation)],
        ]
    if origin is typing.Union:
        sub_types = typing.get_args(annotation)
        if sub_types[-1] is type(None):
            # Union is an Optional type
            if len(sub_types) == 2:
                return tuple(_extract_annotation_signature(sub_types[0]))
            return (
                origin,
                *[_extract_annotation_signature(sub_type) for sub_type in sub_types[:-1]],
            )
        # Union type
        return [
            origin,
            *[_extract_annotation_signature(sub_type) for sub_type in sub_types],
        ]

    raise ValueError(f"Unsupported origin: {origin}")


def _build_json_schema_parameter(arg: Union[list, tuple]) -> Mapping[str, Union[str, Mapping, Sequence]]:
    is_nullable = isinstance(arg, tuple)
    arg_type: Mapping[str, Union[str, Mapping, Sequence]]
    if arg[0] is typing.Union:
        arg_type = {
            "anyOf": [_build_json_schema_parameter(sub_type) for sub_type in arg[1:]],
        }
    if arg[0] is tuple:
        arg_type = {
            "type": "array",
            "items": [_build_json_schema_parameter(sub_type) for sub_type in arg[1:]],
        }
    if arg[0] is list:
        arg_type = {
            "type": "array",
            "items": _build_json_schema_parameter(arg[1]),
        }
    if arg[0] is dict:
        arg_type = {
            "type": "object",
            "properties": {
                "key": _build_json_schema_parameter(arg[1]),
                "value": _build_json_schema_parameter(arg[2]),
            },
        }
    if arg[0] is builtins.str:
        arg_type = {"type": "string"}
    if arg[0] is builtins.int:
        arg_type = {"type": "integer"}
    if arg[0] is builtins.float:
        arg_type = {"type": "number"}
    if arg[0] is builtins.bool:
        arg_type = {"type": "boolean"}

    if is_nullable:
        if arg[0] is typing.Union:
            arg_type["anyOf"] = [  # type: ignore
                {**type_option, "type": [type_option["type"], "null"]}  # type: ignore
                for type_option in arg_type["anyOf"]  # type: ignore
            ]
        else:
            arg_type = {**arg_type, "type": [arg_type["type"], "null"]}

    return arg_type


class JSONSchemaFunctionParameters(TypedDict):
    type: str
    properties: dict[str, dict]
    required: list[str]


def _parameter_is_optional(parameter: inspect.Parameter) -> bool:
    """Check if tool parameter is mandatory."""
    # Check if the parameter can be None, either via Optional[T] or T | None type hint
    origin = typing.get_origin(parameter.annotation)
    # sub_types refers to T inside the annotation
    sub_types = typing.get_args(parameter.annotation)
    return origin is typing.Union and len(sub_types) > 0 and sub_types[-1] is type(None)


def _parse_tool_parameters_schema(func) -> JSONSchemaFunctionParameters:
    properties: dict[str, Any] = {}
    required: list[str] = []
    signature = inspect.signature(func)

    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError(f"{func.__name__}: Varargs and kwargs are not supported by the @tool decorator")

    for parameter in signature.parameters.values():
        try:
            parameter_signature = _extract_annotation_signature(parameter.annotation)
        except ValueError as e:
            raise ValueError(f"{func.__name__}: {parameter.name} lacks a type hint annotation") from e
        param_json_schema = _build_json_schema_parameter(parameter_signature)
        properties[parameter.name] = param_json_schema
        if not _parameter_is_optional(parameter):
            required.append(parameter.name)

    if len(properties) == 0 and len(required) == 0:
        # Edge case: function with no parameters
        return JSONSchemaFunctionParameters(
            type="object",
            properties={},
            required=[],
        )
    return JSONSchemaFunctionParameters(
        type="object",
        # False positive, expected tuple[str] but got tuple[str, ...]
        required=tuple(required),  # type: ignore
        properties=properties,
    )


def _tool_json_schema(func: Callable, strict: bool) -> ToolFunction:
    tool_name = func.__name__
    description = func.__doc__
    if description is None:
        description = ""
    return ToolFunction(
        name=tool_name,
        description=description,
        parameters=_parse_tool_parameters_schema(func),
        strict=strict,
    )


def _build_tool_kernel(
    func: Callable,
    attributes: Optional[dict[str, Optional[Any]]],
    setup_values: Optional[dict[str, Optional[Any]]],
    strict: bool,
) -> ToolKernelRequest:
    return ToolKernelRequest(
        source_code=textwrap.dedent(
            # Remove the tool decorator from source code
            inspect.getsource(func).split("\n", maxsplit=1)[1]
        ),
        attributes=attributes,
        setup_values=setup_values,
        function=_tool_json_schema(
            func=func,
            strict=strict,
        ),
    )


def tool(
    path: Optional[str] = None,
    setup_values: Optional[dict[str, Optional[Any]]] = None,
    attributes: Optional[dict[str, typing.Any]] = None,
    strict: bool = True,
):
    def decorator(func: Callable):
        # Complains about adding attribute on function
        # Nice UX, but mypy doesn't like it
        file_obj = _build_tool_kernel(
            func=func,
            attributes=attributes,
            setup_values=setup_values,
            strict=strict,
        )

        func.json_schema = file_obj.function.model_dump()  # type: ignore

        @wraps(func)
        def wrapper(*args, **kwargs):
            with get_tracer().start_as_current_span(str(uuid.uuid4())) as span:
                trace_metadata = get_trace_context()

                if trace_metadata:
                    write_to_opentelemetry_span(
                        span=span,
                        key=HL_TRACE_METADATA_KEY,
                        value={**trace_metadata, "is_flow_log": False},
                    )
                    push_trace_context(
                        {
                            **trace_metadata,
                            "trace_parent_id": span.get_span_context().span_id,
                            "is_flow_log": False,
                        }
                    )

                output = func(*args, **kwargs)
                if trace_metadata:
                    pop_trace_context()

                tool_log = {
                    "inputs": args_to_inputs(func, args, kwargs),
                }
                if output:
                    tool_log["output"] = output

                write_to_opentelemetry_span(
                    span=span,
                    key=HL_FILE_OT_KEY,
                    value={
                        "path": path if path else func.__name__,
                        "tool": file_obj.model_dump(),
                    },
                )
                write_to_opentelemetry_span(
                    span=span,
                    key=HL_LOG_OT_KEY,
                    value=tool_log,
                )
                return output

        return wrapper

    return decorator
