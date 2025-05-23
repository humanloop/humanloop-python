import builtins
import inspect
import logging
import sys
import textwrap
import typing
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter
from typing import Any, Awaitable, Callable, Literal, Mapping, Optional, Sequence, TypedDict, TypeVar, Union, overload

from opentelemetry.trace import Span, Tracer
from typing_extensions import ParamSpec

from humanloop.context import get_evaluation_context, get_trace_id
from humanloop.decorators.helpers import bind_args
from humanloop.evals import FileEvalConfig
from humanloop.evals.run import HumanloopRuntimeError
from humanloop.otel.constants import (
    HUMANLOOP_FILE_KEY,
    HUMANLOOP_FILE_PATH_KEY,
    HUMANLOOP_FILE_TYPE_KEY,
    HUMANLOOP_LOG_KEY,
)
from humanloop.otel.helpers import process_output, write_to_opentelemetry_span
from humanloop.requests.tool_function import ToolFunctionParams
from humanloop.requests.tool_kernel_request import ToolKernelRequestParams

if sys.version_info >= (3, 10):
    import types

logger = logging.getLogger("humanloop.sdk")


P = ParamSpec("P")
R = TypeVar("R")


def tool_decorator_factory(
    opentelemetry_tracer: Tracer,
    path: str,
    attributes: Optional[dict[str, Any]] = None,
    setup_values: Optional[dict[str, Any]] = None,
):
    def decorator(func: Callable[P, R]) -> Callable[P, Optional[R]]:
        file_type = "tool"

        tool_kernel = _build_tool_kernel(
            func=func,
            attributes=attributes,
            setup_values=setup_values,
            strict=True,
        )

        # Mypy complains about adding attribute on function, but it's nice DX
        func.json_schema = tool_kernel["function"]  # type: ignore

        wrapper = _wrapper_factory(
            opentelemetry_tracer=opentelemetry_tracer,
            func=func,
            path=path,
            tool_kernel=tool_kernel,
            is_awaitable=False,
        )

        wrapper.file = FileEvalConfig(  # type: ignore
            path=path,
            type=file_type,  # type: ignore [arg-type, typeddict-item]
            version=tool_kernel,
            callable=wrapper,
        )

        return wrapper

    return decorator


def a_tool_decorator_factory(
    opentelemetry_tracer: Tracer,
    path: str,
    attributes: Optional[dict[str, Any]] = None,
    setup_values: Optional[dict[str, Any]] = None,
):
    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[Optional[R]]]:
        file_type = "tool"

        tool_kernel = _build_tool_kernel(
            func=func,
            attributes=attributes,
            setup_values=setup_values,
            strict=True,
        )

        # Mypy complains about adding attribute on function, but it's nice DX
        func.json_schema = tool_kernel["function"]  # type: ignore

        wrapper = _wrapper_factory(
            opentelemetry_tracer=opentelemetry_tracer,
            func=func,
            path=path,
            tool_kernel=tool_kernel,
            is_awaitable=True,
        )

        wrapper.file = FileEvalConfig(  # type: ignore
            path=path,
            type=file_type,  # type: ignore [arg-type, typeddict-item]
            version=tool_kernel,
            callable=wrapper,
        )

        return wrapper

    return decorator


@overload
def _wrapper_factory(
    opentelemetry_tracer: Tracer,
    func: Callable[P, Awaitable[R]],
    path: str,
    tool_kernel: ToolKernelRequestParams,
    is_awaitable: Literal[True],
) -> Callable[P, Awaitable[Optional[R]]]: ...


@overload
def _wrapper_factory(
    opentelemetry_tracer: Tracer,
    func: Callable[P, R],
    path: str,
    tool_kernel: ToolKernelRequestParams,
    is_awaitable: Literal[False],
) -> Callable[P, Optional[R]]: ...


def _wrapper_factory(  # type: ignore [misc]
    opentelemetry_tracer: Tracer,
    func: Union[Callable[P, Awaitable[R]], Callable[P, R]],
    path: str,
    tool_kernel: ToolKernelRequestParams,
    is_awaitable: bool,
):
    if is_awaitable:
        func = typing.cast(Callable[P, Awaitable[R]], func)

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Optional[R]:
            evaluation_context = get_evaluation_context()
            if evaluation_context is not None:
                if evaluation_context.path == path:
                    raise HumanloopRuntimeError("Tools cannot be evaluated with the `evaluations.run()` utility.")
            with opentelemetry_tracer.start_as_current_span("humanloop.tool") as span:
                span, log_inputs = _process_inputs(
                    span=span,
                    opentelemetry_tracer=opentelemetry_tracer,
                    path=path,
                    tool_kernel=tool_kernel,
                    func=func,
                    args=args,
                    kwargs=kwargs,
                )

                func_output: Optional[R]
                try:
                    func_output = await func(*args, **kwargs)  # type: ignore [misc]
                    error = None
                except HumanloopRuntimeError as e:
                    # Critical error, re-raise
                    span.record_exception(e)
                    raise e
                except Exception as e:
                    logger.error(f"Error calling {func.__name__}: {e}")
                    error = e
                    func_output = None

                _process_output(
                    span=span,
                    func=func,
                    func_output=func_output,
                    error=error,
                    log_inputs=log_inputs,
                )

                # Return the output of the decorated function
                return func_output
    else:
        func = typing.cast(Callable[P, R], func)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Optional[R]:
            evaluation_context = get_evaluation_context()
            if evaluation_context is not None:
                if evaluation_context.path == path:
                    raise HumanloopRuntimeError("Tools cannot be evaluated with the `evaluations.run()` utility.")
            with opentelemetry_tracer.start_as_current_span("humanloop.tool") as span:
                span, log_inputs = _process_inputs(
                    span=span,
                    opentelemetry_tracer=opentelemetry_tracer,
                    path=path,
                    tool_kernel=tool_kernel,
                    func=func,
                    args=args,
                    kwargs=kwargs,
                )

                func_output: Optional[R]
                try:
                    func_output = func(*args, **kwargs)  # type: ignore [misc]
                    error = None
                except HumanloopRuntimeError as e:
                    # Critical error, re-raise
                    span.record_exception(e)
                    raise e
                except Exception as e:
                    logger.error(f"Error calling {func.__name__}: {e}")
                    error = e
                    func_output = None

                _process_output(
                    span=span,
                    func=func,
                    func_output=func_output,
                    error=error,
                    log_inputs=log_inputs,
                )

                # Return the output of the decorated function
                return func_output

    return wrapper


def _process_inputs(
    span: Span,
    opentelemetry_tracer: Tracer,
    path: str,
    tool_kernel: ToolKernelRequestParams,
    func: Callable[P, R],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Span, dict[str, Any]]:
    """Process inputs before executing the decorated function.

    This handles setting up the OpenTelemetry span and preparing the log inputs.

    Args:
        span: The current OpenTelemetry span
        opentelemetry_tracer: The OpenTelemetry tracer
        path: The path to the tool
        tool_kernel: The tool kernel request parameters
        func: The decorated function
        args: The positional arguments passed to the function
        kwargs: The keyword arguments passed to the function

    Returns:
        A tuple containing the span and the log inputs
    """
    # Write the Tool Kernel to the Span on HL_FILE_OT_KEY
    write_to_opentelemetry_span(
        span=span,  # type: ignore [arg-type]
        key=HUMANLOOP_FILE_KEY,
        value=tool_kernel,  # type: ignore [arg-type]
    )
    span.set_attribute(HUMANLOOP_FILE_PATH_KEY, path)
    span.set_attribute(HUMANLOOP_FILE_TYPE_KEY, "tool")

    log_inputs: dict[str, Any] = bind_args(func, args, kwargs)

    return span, log_inputs


def _process_output(
    span: Span,
    func: Union[Callable[P, R], Callable[P, Awaitable[R]]],
    func_output: Optional[R],
    error: Optional[Exception],
    log_inputs: dict[str, Any],
) -> None:
    """Process outputs after executing the decorated function.

    This handles processing the function output, error logging, and writing to the OpenTelemetry span.

    Args:
        span: The current OpenTelemetry span
        func: The decorated function
        func_output: The output from the function execution
        error: Any exception that occurred during function execution
        log_inputs: The input parameters logged during processing
    """
    log_error: Optional[str]
    log_output: str

    if not error:
        log_output = process_output(
            func=func,
            output=func_output,
        )
        log_error = None
    else:
        output = None
        log_output = process_output(
            func=func,
            output=output,
        )
        log_error = str(error)

    # Populate Tool Log attributes
    tool_log = {
        "inputs": log_inputs,
        "output": log_output,
        "error": log_error,
        "trace_parent_id": get_trace_id(),
    }
    # Write the Tool Log to the Span on HL_LOG_OT_KEY
    write_to_opentelemetry_span(
        span=span,  # type: ignore [arg-type]
        key=HUMANLOOP_LOG_KEY,
        value=tool_log,  # type: ignore [arg-type]
    )


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
