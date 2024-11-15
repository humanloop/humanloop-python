import builtins
import inspect
import logging
import sys
import textwrap
import typing
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, TypedDict, Union

from opentelemetry.trace import Tracer
from typing_extensions import Unpack

from humanloop.decorators.helpers import args_to_inputs
from humanloop.eval_utils import File
from humanloop.otel import TRACE_FLOW_CONTEXT, FlowContext
from humanloop.otel.constants import (
    HUMANLOOP_FILE_KEY,
    HUMANLOOP_FILE_TYPE_KEY,
    HUMANLOOP_LOG_KEY,
    HUMANLOOP_PATH_KEY,
)
from humanloop.otel.helpers import generate_span_id, jsonify_if_not_string, write_to_opentelemetry_span
from humanloop.requests.tool_function import ToolFunctionParams
from humanloop.requests.tool_kernel_request import ToolKernelRequestParams

if sys.version_info >= (3, 10):
    import types

logger = logging.getLogger("humanloop.sdk")


def tool(
    opentelemetry_tracer: Tracer,
    path: Optional[str] = None,
    **tool_kernel: Unpack[ToolKernelRequestParams],  # type: ignore
):
    def decorator(func: Callable):
        enhanced_tool_kernel = _build_tool_kernel(
            func=func,
            attributes=tool_kernel.get("attributes"),
            setup_values=tool_kernel.get("setup_values"),
            strict=True,
        )

        # Mypy complains about adding attribute on function, but it's nice UX
        func.json_schema = enhanced_tool_kernel["function"]  # type: ignore

        @wraps(func)
        def wrapper(*args, **kwargs):
            with opentelemetry_tracer.start_as_current_span(generate_span_id()) as span:
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
                span.set_attribute(HUMANLOOP_PATH_KEY, path if path else func.__name__)
                span.set_attribute(HUMANLOOP_FILE_TYPE_KEY, "tool")
                if enhanced_tool_kernel:
                    write_to_opentelemetry_span(
                        span=span,
                        key=f"{HUMANLOOP_FILE_KEY}.tool",
                        value=enhanced_tool_kernel,
                    )

                # Call the decorated function
                try:
                    output = func(*args, **kwargs)
                    output_stringified = jsonify_if_not_string(
                        func=func,
                        output=output,
                    )
                    error = None
                except Exception as e:
                    logger.error(f"Error calling {func.__name__}: {e}")
                    output = None
                    output_stringified = jsonify_if_not_string(
                        func=func,
                        output=output,
                    )
                    error = str(e)

                # Populate known Tool Log attributes
                tool_log = {
                    "inputs": args_to_inputs(func, args, kwargs),
                    "output": output_stringified,
                    "error": error,
                }

                # Write the Tool Log to the Span on HL_LOG_OT_KEY
                write_to_opentelemetry_span(
                    span=span,
                    key=HUMANLOOP_LOG_KEY,
                    value=tool_log,
                )

                # Return the output of the decorated function
                return output

        wrapper.file = File(  # type: ignore
            path=path if path else func.__name__,
            type="tool",
            version=enhanced_tool_kernel,
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
    try:
        source_code = textwrap.dedent(inspect.getsource(func))
    except TypeError as e:
        raise TypeError(
            f"Cannot extract source code for function {func.__name__}. "
            "Try decorating a plain function instead of a partial for example."
        ) from e
    # Remove decorator from source code by finding first 'def'
    # This makes the source_code extraction idempotent whether
    # the decorator is applied directly or used as a higher-order
    # function
    source_code = source_code[source_code.find("def") :]
    kernel = ToolKernelRequestParams(
        source_code=source_code,
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
            raise ValueError(f"{func.__name__}: *args and **kwargs are not supported by the @tool decorator")

    for parameter in signature.parameters.values():
        try:
            parameter_signature = _parse_annotation(parameter.annotation)
        except ValueError as e:
            raise ValueError(f"Error parsing signature of @tool annotated function {func.__name__}: {e}") from e
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


if sys.version_info >= (3, 11):
    _PRIMITIVE_TYPES = Union[
        str,
        int,
        float,
        bool,
        Parameter.empty,  # type: ignore
        Ellipsis,  # type: ignore
    ]
else:
    # Ellipsis not supported as type before Python 3.11
    _PRIMITIVE_TYPES = Union[
        str,
        int,
        float,
        bool,
        Parameter.empty,  # type: ignore
    ]


@dataclass
class _ParsedAnnotation:
    def no_type_hint(self) -> bool:
        """Check if the annotation has no type hint.

        Examples:
            str -> False
            list -> True
            list[str] -> False
        """
        raise NotImplementedError


@dataclass
class _ParsedPrimitiveAnnotation(_ParsedAnnotation):
    annotation: _PRIMITIVE_TYPES

    def no_type_hint(self) -> bool:
        return self.annotation is Parameter.empty or self.annotation is Ellipsis


@dataclass
class _ParsedDictAnnotation(_ParsedAnnotation):
    # Both are null if no type hint e.g. dict vs dict[str, int]
    key_annotation: Optional[_ParsedAnnotation]
    value_annotation: Optional[_ParsedAnnotation]

    def no_type_hint(self) -> bool:
        return self.key_annotation is None and self.value_annotation is None


@dataclass
class _ParsedTupleAnnotation(_ParsedAnnotation):
    # Null if no type hint e.g. tuple vs tuple[str, int]
    annotation: Optional[list[_ParsedAnnotation]]

    def no_type_hint(self) -> bool:
        return self.annotation is None


@dataclass
class _ParsedUnionAnnotation(_ParsedAnnotation):
    annotation: list[_ParsedAnnotation]


@dataclass
class _ParsedListAnnotation(_ParsedAnnotation):
    # Null if no type hint e.g. list vs list[str]
    annotation: Optional[_ParsedAnnotation]


@dataclass
class _ParsedOptionalAnnotation(_ParsedAnnotation):
    annotation: _ParsedAnnotation


def _parse_annotation(annotation: typing.Type) -> _ParsedAnnotation:
    """Parse constituent parts of a potentially nested type hint.

    Custom types are not supported, only built-in types and typing module types.

    """
    origin = typing.get_origin(annotation)
    if origin is None:
        # Either not a nested type or no type hint
        # Parameter.empty is used for parameters without type hints
        # Ellipsis is interpreted as Any
        if annotation not in (
            str,
            int,
            float,
            bool,
            Parameter.empty,
            Ellipsis,
            dict,
            list,
            tuple,
        ):
            raise ValueError(f"Unsupported type hint: {annotation}")

        # Check if it's a complex type with no inner type
        if annotation == builtins.dict:
            return _ParsedDictAnnotation(
                value_annotation=None,
                key_annotation=None,
            )
        if annotation == builtins.list:
            return _ParsedListAnnotation(
                annotation=None,
            )
        if annotation == builtins.tuple:
            return _ParsedTupleAnnotation(
                annotation=None,
            )

        # Is a primitive type
        return _ParsedPrimitiveAnnotation(
            annotation=annotation,
        )

    if origin is list:
        inner_annotation = _parse_annotation(typing.get_args(annotation)[0])
        return _ParsedListAnnotation(
            annotation=inner_annotation,
        )

    if origin is dict:
        key_type = _parse_annotation(typing.get_args(annotation)[0])
        value_type = _parse_annotation(typing.get_args(annotation)[1])
        return _ParsedDictAnnotation(
            key_annotation=key_type,
            value_annotation=value_type,
        )

    if origin is tuple:
        return _ParsedTupleAnnotation(
            annotation=[_parse_annotation(arg) for arg in typing.get_args(annotation)],
        )

    if origin is typing.Union or (sys.version_info >= (3, 10) and origin is types.UnionType):
        sub_types = typing.get_args(annotation)
        if sub_types[-1] is type(None):
            # type(None) in sub_types indicates Optional type
            if len(sub_types) == 2:
                # Union is an Optional type only
                return _ParsedOptionalAnnotation(
                    annotation=_parse_annotation(sub_types[0]),
                )
            # Union has sub_types and is Optional
            return _ParsedOptionalAnnotation(
                annotation=_ParsedUnionAnnotation(
                    annotation=[_parse_annotation(sub_type) for sub_type in sub_types[:-1]],
                )
            )
        # Union type that is not Optional
        return _ParsedUnionAnnotation(
            annotation=[_parse_annotation(sub_type) for sub_type in sub_types],
        )

    raise ValueError(f"Unsupported origin: {origin}")


_JSON_SCHEMA_ANY = ["string", "integer", "number", "boolean", "object", "array", "null"]


def _annotation_parse_to_json_schema(
    arg: _ParsedAnnotation,
) -> Mapping[str, Union[str, Mapping, Sequence]]:
    """
    Convert parse result from _parse_annotation to JSON Schema for a parameter.

    The function recursively converts the nested type hints to a JSON Schema.

    Note that 'any' is not supported by JSON Schema, so we allow any type as a workaround.
    """
    arg_type: Mapping[str, Union[str, Mapping, Sequence]]

    if isinstance(arg, _ParsedOptionalAnnotation):
        is_optional = True
        arg = arg.annotation
    else:
        is_optional = False

    if isinstance(arg, _ParsedUnionAnnotation):
        arg_type = {
            "anyOf": [_annotation_parse_to_json_schema(sub_type) for sub_type in arg.annotation],
        }

    elif isinstance(arg, _ParsedTupleAnnotation):
        if arg.annotation is None:
            # tuple annotation with no type hints
            # This is equivalent with a list, since the
            # number of elements is not specified
            arg_type = {
                "type": "array",
                "items": {"type": _JSON_SCHEMA_ANY},
            }
        else:
            arg_type = {
                "type": "array",
                "items": [_annotation_parse_to_json_schema(sub_type) for sub_type in arg.annotation],
            }

    elif isinstance(arg, _ParsedListAnnotation):
        if arg.annotation is None:
            # list annotation with no type hints
            if is_optional:
                arg_type = {
                    "type": ["array", "null"],
                    "items": {"type": _JSON_SCHEMA_ANY},
                }
            else:
                arg_type = {
                    "type": "array",
                    "items": {"type": _JSON_SCHEMA_ANY},
                }
        else:
            arg_type = {
                "type": "array",
                "items": _annotation_parse_to_json_schema(arg.annotation),
            }

    elif isinstance(arg, _ParsedDictAnnotation):
        if arg.key_annotation is None and arg.value_annotation is None:
            # dict annotation with no type hints
            if is_optional:
                arg_type = {
                    "type": ["object", "null"],
                    "properties": {
                        "key": {"type": _JSON_SCHEMA_ANY},
                        "value": {"type": _JSON_SCHEMA_ANY},
                    },
                }
            else:
                arg_type = {
                    "type": "object",
                    "properties": {
                        "key": {"type": _JSON_SCHEMA_ANY},
                        "value": {"type": _JSON_SCHEMA_ANY},
                    },
                }
        else:
            arg_type = {
                "type": "object",
                "properties": {
                    "key": _annotation_parse_to_json_schema(arg.key_annotation),  # type: ignore
                    "value": _annotation_parse_to_json_schema(arg.value_annotation),  # type: ignore
                },
            }

    elif isinstance(arg, _ParsedPrimitiveAnnotation):
        if arg.annotation is builtins.str:
            arg_type = {"type": "string"}
        if arg.annotation is builtins.int:
            arg_type = {"type": "integer"}
        if arg.annotation is builtins.float:
            arg_type = {"type": "number"}
        if arg.annotation is builtins.bool:
            arg_type = {"type": "boolean"}
        if arg.annotation is Parameter.empty or arg.annotation is Ellipsis:
            # JSON Schema dropped support for 'any' type, we allow any type as a workaround
            arg_type = {"type": _JSON_SCHEMA_ANY}

    else:
        raise ValueError(f"Unsupported annotation type: {arg}")

    if is_optional:
        if isinstance(arg, _ParsedUnionAnnotation):
            for type_option in arg_type["anyOf"]:
                if (
                    isinstance(type_option["type"], list)  # type: ignore
                    and "null" not in type_option["type"]  # type: ignore
                ):  # type: ignore
                    type_option["type"] = [*type_option["type"], "null"]  # type: ignore
                elif not isinstance(type_option["type"], list):  # type: ignore
                    type_option["type"] = [type_option["type"], "null"]  # type: ignore
        else:
            if isinstance(arg_type["type"], list) and "null" not in arg_type["type"]:  # type: ignore
                arg_type = {**arg_type, "type": [*arg_type["type"], "null"]}  # type: ignore
            elif not isinstance(arg_type["type"], list):  # type: ignore
                arg_type = {**arg_type, "type": [arg_type["type"], "null"]}  # type: ignore

    return arg_type


def _parameter_is_optional(
    parameter: inspect.Parameter,
) -> bool:
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
    return (
        (origin is typing.Union or (sys.version_info >= (3, 10) and origin is types.UnionType))
        and len(sub_types) > 0
        and sub_types[-1] is type(None)
    )
