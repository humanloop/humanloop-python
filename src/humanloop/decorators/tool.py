import builtins
import inspect
import textwrap
import typing
import uuid
from functools import wraps
from inspect import Parameter
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, TypedDict, Union

from opentelemetry.trace import Tracer

from humanloop.eval_utils import File
from humanloop.otel import TRACE_FLOW_CONTEXT, FlowContext
from humanloop.otel.constants import HL_FILE_KEY, HL_FILE_TYPE_KEY, HL_LOG_KEY, HL_PATH_KEY
from humanloop.otel.helpers import write_to_opentelemetry_span
from humanloop.requests.tool_function import ToolFunctionParams
from humanloop.requests.tool_kernel_request import ToolKernelRequestParams

from .helpers import args_to_inputs


def tool(
    opentelemetry_tracer: Tracer,
    path: Optional[str] = None,
    setup_values: Optional[dict[str, Optional[Any]]] = None,
    attributes: Optional[dict[str, typing.Any]] = None,
    strict: bool = True,
):
    def decorator(func: Callable):
        tool_kernel = _build_tool_kernel(
            func=func,
            attributes=attributes,
            setup_values=setup_values,
            strict=strict,
        )

        # Mypy complains about adding attribute on function, but it's nice UX
        func.json_schema = tool_kernel["function"]  # type: ignore

        @wraps(func)
        def wrapper(*args, **kwargs):
            with opentelemetry_tracer.start_as_current_span(str(uuid.uuid4())) as span:
                span_id = span.get_span_context().span_id
                if span.parent:
                    span_parent_id = span.parent.span_id
                else:
                    span_parent_id = None
                parent_trace_metadata = TRACE_FLOW_CONTEXT.get(span_parent_id)
                if parent_trace_metadata:
                    TRACE_FLOW_CONTEXT[span_id] = FlowContext(
                        span_id=span_id,
                        trace_parent_id=span_parent_id,
                        is_flow_log=False,
                    )

                # Write the Tool Kernel to the Span on HL_FILE_OT_KEY
                span.set_attribute(HL_PATH_KEY, path if path else func.__name__)
                span.set_attribute(HL_FILE_TYPE_KEY, "tool")
                if tool_kernel:
                    write_to_opentelemetry_span(
                        span=span,
                        key=f"{HL_FILE_KEY}.tool",
                        value=tool_kernel,
                    )

                # Call the decorated function
                output = func(*args, **kwargs)

                # Populate known Tool Log attributes
                tool_log = {
                    "inputs": args_to_inputs(func, args, kwargs),
                }
                if output:
                    tool_log["output"] = output

                # Write the Tool Log to the Span on HL_LOG_OT_KEY
                if tool_log:
                    write_to_opentelemetry_span(
                        span=span,
                        key=HL_LOG_KEY,
                        value=tool_log,
                    )

                # Return the output of the decorated function
                return output

        func.file = File(  # type: ignore
            path=path if path else func.__name__,
            type="tool",
            version=tool_kernel,
            is_decorated=True,
            callable=wrapper,
        )

        return wrapper

    return decorator


def _build_tool_kernel(
    func: Callable,
    attributes: Optional[dict[str, Optional[Any]]],
    setup_values: Optional[dict[str, Optional[Any]]],
    strict: bool,
) -> ToolKernelRequestParams:
    """Build ToolKernelRequest object from decorated function."""
    kernel = ToolKernelRequestParams(
        source_code=textwrap.dedent(
            # Remove the tool decorator from source code
            inspect.getsource(func).split("\n", maxsplit=1)[1]
        ),
        # Note: OTel complains about falsy values in attributes, so we use OT_EMPTY_ATTRIBUTE
        function=_build_function_property(
            func=func,
            strict=strict,
        ),
    )
    if attributes:
        kernel["attributes"] = attributes
    if setup_values:
        kernel["setup_values"] = setup_values
    return kernel


def _build_function_property(func: Callable, strict: bool) -> ToolFunctionParams:
    """Build `function` property inside ToolKernelRequest."""
    tool_name = func.__name__
    description = func.__doc__
    if description is None:
        description = ""
    return ToolFunctionParams(
        name=tool_name,
        description=description,
        parameters=_build_function_parameters_property(func),  # type: ignore
        strict=strict,
    )


class _JSONSchemaFunctionParameters(TypedDict):
    type: str
    properties: dict[str, typing.Union[dict, list]]
    required: list[str]
    additionalProperties: Literal[False]


def _build_function_parameters_property(func) -> _JSONSchemaFunctionParameters:
    """Build `function.parameters` property inside ToolKernelRequest."""
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
            parameter_signature = _parse_annotation(parameter.annotation)
        except ValueError as e:
            raise ValueError(f"{func.__name__}: {e.args[0]}") from e
        param_json_schema = _annotation_parse_to_json_schema(parameter_signature)
        properties[parameter.name] = param_json_schema
        if not _parameter_is_optional(parameter):
            required.append(parameter.name)

    if len(properties) == 0 and len(required) == 0:
        # Edge case: function with no parameters
        return _JSONSchemaFunctionParameters(
            type="object",
            properties={},
            required=[],
            additionalProperties=False,
        )
    return _JSONSchemaFunctionParameters(
        type="object",
        # False positive, expected tuple[str] but got tuple[str, ...]
        required=tuple(required),  # type: ignore
        properties=properties,
        additionalProperties=False,
    )


def _parse_annotation(annotation: typing.Type) -> Union[list, tuple]:
    """Parse constituent parts of a potentially nested type hint.

    Custom types are not supported, only built-in types and typing module types.


    Method returns potentially nested lists, with each list describing a
    level of type nesting. For a nested type, the function recursively calls
    itself to parse the inner type.

    When the annotation is optional, a tuple is returned with the inner type
    to signify that the parameter is nullable.

    For lists, a list with two elements is returned, where the first element
    is the list type and the second element is the inner type.

    For dictionaries, a list with three elements is returned, where the first
    element is the dict type, the second element is the key type, and the
    third element is the value type.

    For tuples, a list where the fist element is the tuple type and the rest
    describes the inner types.

    For Union types, a list with the first element being the Union type and
    the rest describing the inner types.

    Note that for nested types that lack inner type, e.g. list instead of
    list[str], the inner type is set to Parameter.empty. This edge case is
    handled by _annotation_parse_to_json_schema.

    Examples:
        str -> [str]
        Optional[str] -> (str)
        str | None -> (str)

        list[str] -> [list, [str]]
        Optional[list[str]] -> (list, [str])

        dict[str, int] -> [dict, [str], [int]]
        Optional[dict[str, int]] -> (dict, [str], [int])

        list[list[str]] -> [list, [list, str]]
        list[Optional[list[str]]] -> [list, (list, [str])]

        dict[str, Optional[int]] -> [dict, [str], (int)]

        Union[str, int] -> [Union, [str], [int]]

        tuple[str, int, list[str]] -> [tuple, [str], [int], [list, str]]
        tuple[Optional[str], int, Optional[list[str]]] -> (str, [int], (list, str))

        list -> [list]
    """
    origin = typing.get_origin(annotation)
    if origin is None:
        # Either not a nested type or no type hint
        # Parameter.empty is used for parameters without type hints
        if annotation not in (str, int, float, bool, Parameter.empty, dict, list, tuple):
            raise ValueError(f"Unsupported type hint: {annotation}")
        return [annotation]
    if origin is list:
        inner_type = _parse_annotation(typing.get_args(annotation)[0])
        return [origin, inner_type]
    if origin is dict:
        key_type = _parse_annotation(typing.get_args(annotation)[0])
        value_type = _parse_annotation(typing.get_args(annotation)[1])
        return [origin, key_type, value_type]
    if origin is tuple:
        return [
            origin,
            *[_parse_annotation(arg) for arg in typing.get_args(annotation)],
        ]
    if origin is typing.Union:
        sub_types = typing.get_args(annotation)
        if sub_types[-1] is type(None):
            # Union is an Optional type
            if len(sub_types) == 2:
                return tuple(_parse_annotation(sub_types[0]))
            return (
                origin,
                *[_parse_annotation(sub_type) for sub_type in sub_types[:-1]],
            )
        # Union type
        return [
            origin,
            *[_parse_annotation(sub_type) for sub_type in sub_types],
        ]

    raise ValueError(f"Unsupported origin: {origin}")


def _annotation_parse_to_json_schema(arg: Union[list, tuple]) -> Mapping[str, Union[str, Mapping, Sequence]]:
    """
    Convert parse result from _parse_annotation to JSON Schema for a parameter.

    The function recursively converts the nested type hints to a JSON Schema.

    Note that 'any' is not supported by JSON Schema, so we allow any type as a workaround.

    Examples:
        [str] -> {"type": "string"}
        (str) -> {"type": ["string", "null"]}

        [list, [str]] -> {"type": "array", "items": {"type": "string"}}
        (list, [str]) -> {"type": ["array", "null"], "items": {"type": "string"}}

        [dict, [str], [int]] ->
            {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "integer"}
                }
            }

        [list, [list, str]] ->
            {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }

        tuple[str, int, list[str]] ->
            {
                type: "array",
                items: [
                    {"type": "string"},
                    {"type": "integer"},
                    {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                ]
            }

        Union[str, int] ->
            {
                "anyOf": [
                    {"type": "string"},
                    {"type": "integer"}
                ]
            }

        dict[int, list] ->
            {
                "type": "object",
                "properties": {
                    "key": {"type": "integer"},
                    "value": {
                        "type": "array",
                        "items": {"type": ["string", "integer", "number", "boolean", "object", "array", "null"]}
                    }
                }
            }

        Optional[list] ->
            {
                "type": ["array", "null"],
                "items": {"type": ["string", "integer", "number", "boolean", "object", "array", "null"]},
            }
    """
    is_nullable = isinstance(arg, tuple)
    arg_type: Mapping[str, Union[str, Mapping, Sequence]]
    if arg[0] is typing.Union:
        arg_type = {
            "anyOf": [_annotation_parse_to_json_schema(sub_type) for sub_type in arg[1:]],
        }
    if arg[0] is tuple:
        if len(arg) == 1:
            # tuple annotation with no type hints
            # This is equivalent with a list, since the
            # number of elements is not specified
            arg_type = {
                "type": "array",
                "items": {"type": ["string", "integer", "number", "boolean", "object", "array", "null"]},
            }
        else:
            arg_type = {
                "type": "array",
                "items": [_annotation_parse_to_json_schema(sub_type) for sub_type in arg[1:]],
            }
    if arg[0] is list:
        if len(arg) == 1:
            # list annotation with no type hints
            if isinstance(arg, tuple):
                # Support Optional annotation
                arg = (list, [Parameter.empty])
            else:
                # Support non-Optional list annotation
                arg = [list, [Parameter.empty]]
        arg_type = {
            "type": "array",
            "items": _annotation_parse_to_json_schema(arg[1]),
        }
    if arg[0] is dict:
        if len(arg) == 1:
            # dict annotation with no type hints
            if isinstance(arg, tuple):
                arg = (dict, [Parameter.empty], [Parameter.empty])
            else:
                arg = [dict, [Parameter.empty], [Parameter.empty]]
        arg_type = {
            "type": "object",
            "properties": {
                "key": _annotation_parse_to_json_schema(arg[1]),
                "value": _annotation_parse_to_json_schema(arg[2]),
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
    if arg[0] is Parameter.empty:
        # JSON Schema dropped support for 'any' type, we allow any type as a workaround
        arg_type = {"type": ["string", "integer", "number", "boolean", "object", "array", "null"]}

    if is_nullable:
        if arg[0] is typing.Union:
            arg_type["anyOf"] = [  # type: ignore
                {**type_option, "type": [type_option["type"], "null"]}  # type: ignore
                for type_option in arg_type["anyOf"]  # type: ignore
            ]
        else:
            arg_type = {**arg_type, "type": [arg_type["type"], "null"]}

    return arg_type


def _parameter_is_optional(parameter: inspect.Parameter) -> bool:
    """Check if tool parameter is mandatory.

    Examples:
        Optional[T] -> True
        T | None -> True
        T -> False
    """
    # Check if the parameter can be None, either via Optional[T] or T | None type hint
    origin = typing.get_origin(parameter.annotation)
    # sub_types refers to T inside the annotation
    sub_types = typing.get_args(parameter.annotation)
    return origin is typing.Union and len(sub_types) > 0 and sub_types[-1] is type(None)
