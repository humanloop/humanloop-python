import builtins
import inspect
import textwrap
import typing
import uuid
from functools import wraps


from humanloop.otel import get_trace_context, get_tracer, pop_trace_context, push_trace_context
from humanloop.otel.constants import HL_FILE_OT_KEY, HL_LOG_OT_KEY, HL_TRACE_METADATA_KEY
from humanloop.otel.helpers import write_to_opentelemetry_span

from .helpers import args_to_inputs


def _type_to_schema(type_hint):
    match type_hint:
        case builtins.int:
            return "number"
        case builtins.float:
            return "number"
        case builtins.bool:
            return "boolean"
        case builtins.str:
            return "string"
        case builtins.dict:
            return "object"
        case _:
            raise ValueError(f"Unsupported type hint: {type_hint}")


def _handle_dict_annotation(parameter: inspect.Parameter) -> dict[str, object]:
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
    return {
        "type": "object",
        "properties": {
            "key": {"type": _type_to_schema(type_key)},
            "value": {"type": _type_to_schema(type_value)},
        },
    }


def _handle_list_annotation(parameter: inspect.Parameter) -> dict[str, object]:
    try:
        list_type = typing.get_args(parameter.annotation)[0]
    except ValueError:
        raise ValueError("List annotation must have one type hint")
    return {
        "type": "array",
        "items": {"type": _type_to_schema(list_type)},
    }


def _handle_union_annotation(parameter: inspect.Parameter) -> dict[str, object]:
    union_types = [sub_type for sub_type in typing.get_args(parameter.annotation) if sub_type != type(None)]
    if len(union_types) != 1:
        raise ValueError("Union types are not supported. Try passing a string and parsing inside function")
    return {"type": _type_to_schema(union_types[0])}


def _handle_simple_type(parameter: inspect.Parameter) -> dict[str, object]:
    if parameter.annotation is None:
        raise ValueError("Parameters must have type hints")
    return {"type": _type_to_schema(parameter.annotation)}


def _parse_tool_parameters_schema(func) -> dict[str, dict]:
    # TODO: Add tests for this, 100% it is breakable
    signature = inspect.signature(func)
    required = []
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError("Varargs and kwargs are not supported")
        match typing.get_origin(parameter.annotation):
            case builtins.dict:
                param_schema = _handle_dict_annotation(parameter)
                parameters_schema["required"].append(parameter.name)
                required.append(parameter.name)
            case builtins.list:
                param_schema = _handle_list_annotation(parameter)
                parameters_schema["required"].append(parameter.name)
                required.append(parameter.name)
            case typing.Union:
                param_schema = _handle_union_annotation(parameter)
            case None:
                param_schema = _handle_simple_type(parameter)
                required.append(parameter.name)
            case _:
                raise ValueError("Unsupported type hint ", parameter)
        parameters_schema["properties"][parameter.name] = param_schema
        parameters_schema["required"] = required
    return parameters_schema


def _tool_json_schema(func: callable):
    tool_name = func.__name__
    description = func.__doc__
    if description is None:
        description = ""
    return {
        "name": tool_name,
        "description": description,
        "parameters": _parse_tool_parameters_schema(func),
    }


def _extract_tool_kernel(func: callable) -> dict:
    return {
        "source_code": textwrap.dedent(
            # Remove the tool decorator from source code
            inspect.getsource(func).split("\n", maxsplit=1)[1]
        ),
        "function": _tool_json_schema(func),
        "tool_type": "json_schema",
        "strict": True,
    }


def tool(path: str | None = None, attributes: dict[str, typing.Any] | None = None):
    """Decorator to mark a function as a Humanloop Tool.

    The decorator inspect the wrapped function signature and code to infer
    the File kernel and JSON schema for the Tool. Any change to the decorated
    function will create a new version of the Tool, provided that the path
    remains the same.

    Every call to the decorated function will create a Log against the Tool.

    Arguments:
        path: Optional. The path to the Tool. If not provided, the function name
            will be used as the path and the File will be created in the root
            of your Humanloop's organization workspace.
    """

    def decorator(func: callable):
        func.json_schema = _tool_json_schema(func)
        decorator.__hl_file_id = uuid.uuid4()

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
