import builtins
import inspect
import textwrap
import typing
import uuid
from functools import wraps
from typing import Callable, Literal, Optional, TypedDict, Union

from humanloop.otel import get_trace_context, get_tracer, pop_trace_context, push_trace_context
from humanloop.otel.constants import HL_FILE_OT_KEY, HL_LOG_OT_KEY, HL_TRACE_METADATA_KEY
from humanloop.otel.helpers import write_to_opentelemetry_span

from .helpers import args_to_inputs


class JSONSchemaProperty(TypedDict):
    type: Literal["number", "boolean", "string", "object"]


class JSONSchemaArray(TypedDict):
    type: Literal["array"]
    items: JSONSchemaProperty


class JSONSchemaObjectProperty(TypedDict):
    key: JSONSchemaProperty
    value: JSONSchemaProperty


class JSONSchemaObject(TypedDict):
    type: Literal["object"]
    properties: JSONSchemaObjectProperty


class JSONSchemaFunctionParameters(TypedDict):
    type: Literal["object"]
    required: tuple[str]
    properties: dict[str, Union[JSONSchemaProperty, JSONSchemaArray, JSONSchemaObject]]


def _type_to_schema(type_hint):
    if type_hint is int:
        return "number"
    if type_hint is float:
        return "number"
    if type_hint is bool:
        return "boolean"
    if type_hint is str:
        return "string"
    if type_hint is dict:
        return "object"
    raise ValueError(f"Unsupported type hint: {type_hint}")


def _handle_dict_annotation(parameter: inspect.Parameter) -> JSONSchemaObject:
    try:
        type_key, type_value = typing.get_args(parameter.annotation)
    except ValueError:
        raise ValueError("Dict annotation must have two type hints")
    if type_key not in (builtins.str, builtins.int, typing.Literal, builtins.float):
        raise ValueError("Dict keys must be strings or integers", parameter.name, type_key)
    if type_value not in (
        builtins.str,
        builtins.int,
        typing.Literal,
        builtins.float,
        dict,
    ):
        raise ValueError("Dict values must be strings or integers", parameter.name, type_value)
    return JSONSchemaObject(
        type="object",
        properties=JSONSchemaObjectProperty(
            key={"type": _type_to_schema(type_key)},
            value={"type": _type_to_schema(type_value)},
        ),
    )


def _handle_list_annotation(parameter: inspect.Parameter) -> JSONSchemaArray:
    try:
        list_type = typing.get_args(parameter.annotation)[0]
    except ValueError:
        raise ValueError("List annotation must have one type hint")
    return JSONSchemaArray(
        type="array",
        items={
            "type": _type_to_schema(list_type),
        },
    )


def _handle_optional_annotation(parameter: inspect.Parameter) -> JSONSchemaProperty:
    union_types = [sub_type for sub_type in typing.get_args(parameter.annotation) if sub_type != type(None)]
    if len(union_types) != 1:
        raise ValueError("Union types are not supported. Try passing a string and parsing inside function")
    return {"type": _type_to_schema(union_types[0])}


def _handle_simple_type(parameter: inspect.Parameter) -> JSONSchemaProperty:
    if parameter.annotation is None:
        raise ValueError("Parameters must have type hints")
    return {"type": _type_to_schema(parameter.annotation)}


def _parse_tool_parameters_schema(func) -> JSONSchemaFunctionParameters:
    # TODO: Add tests for this, 100% it is breakable
    signature = inspect.signature(func)
    required: list[str] = []
    properties: dict[str, Union[JSONSchemaArray, JSONSchemaProperty, JSONSchemaObject]] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError("Varargs and kwargs are not supported")
        origin = typing.get_origin(parameter.annotation)
        param_schema: Union[JSONSchemaProperty, JSONSchemaArray, JSONSchemaObject]
        if origin is Union:
            param_schema = _handle_optional_annotation(parameter)
        elif origin is None:
            param_schema = _handle_simple_type(parameter)
            required.append(parameter.name)
        elif isinstance(origin, dict):
            param_schema = _handle_dict_annotation(parameter)
            required.append(parameter.name)
        elif isinstance(origin, list):
            param_schema = _handle_list_annotation(parameter)
            required.append(parameter.name)
        else:
            raise ValueError("Unsupported type hint ", parameter)
        properties[parameter.name] = param_schema
    return JSONSchemaFunctionParameters(
        type="object",
        # False positive, expected tuple[str] but got tuple[str, ...]
        required=tuple(required),  # type: ignore
        properties=properties,
    )


class JSONSchemaFunction(TypedDict):
    name: str
    description: str
    parameters: JSONSchemaFunctionParameters


def _tool_json_schema(func: Callable) -> JSONSchemaFunction:
    tool_name = func.__name__
    description = func.__doc__
    if description is None:
        description = ""
    return JSONSchemaFunction(
        name=tool_name,
        description=description,
        parameters=_parse_tool_parameters_schema(func),
    )


class ToolKernel(TypedDict):
    source_code: str
    function: JSONSchemaFunction
    tool_type: Literal["json_schema"]
    strict: Literal[True]


def _extract_tool_kernel(func: Callable) -> ToolKernel:
    return ToolKernel(
        source_code=textwrap.dedent(
            # Remove the tool decorator from source code
            inspect.getsource(func).split("\n", maxsplit=1)[1]
        ),
        function=_tool_json_schema(func=func),
        tool_type="json_schema",
        strict=True,
    )


def tool(path: Optional[str] = None, attributes: Optional[dict[str, typing.Any]] = None):
    def decorator(func: Callable):
        # Complains about adding attribute on function
        # Nice UX, but mypy doesn't like it
        func.json_schema = _tool_json_schema(func)  # type: ignore

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
                        "tool": {
                            **_extract_tool_kernel(func),
                            "attributes": attributes,
                        },
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
