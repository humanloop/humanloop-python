from typing import Any, Optional, TypedDict, Union

import pytest
from humanloop.decorators.tool import tool
from humanloop.otel.constants import HUMANLOOP_FILE_KEY, HUMANLOOP_LOG_KEY
from humanloop.otel.helpers import read_from_opentelemetry_span
from jsonschema.protocols import Validator
from opentelemetry.sdk.trace import Tracer
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_calculator_decorator(
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    # GIVEN a test OpenTelemetry configuration
    tracer, exporter = opentelemetry_test_configuration

    @tool(opentelemetry_tracer=tracer)
    def calculator(operation: str, num1: float, num2: float) -> float:
        """Do arithmetic operations on two numbers."""
        if operation == "add":
            return num1 + num2
        elif operation == "subtract":
            return num1 - num2
        elif operation == "multiply":
            return num1 * num2
        elif operation == "divide":
            return num1 / num2
        else:
            raise ValueError(f"Invalid operation: {operation}")

    # WHEN calling the @tool decorated function
    result = calculator(operation="add", num1=1, num2=2)
    assert result == 3
    # THEN a single span is created and the log and file attributes are correctly set
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    hl_file: dict[str, Any] = read_from_opentelemetry_span(span=spans[0], key=HUMANLOOP_FILE_KEY)
    hl_log: dict[str, Any] = read_from_opentelemetry_span(span=spans[0], key=HUMANLOOP_LOG_KEY)
    assert hl_log["output"] == str(result) == "3"
    assert hl_log["inputs"] == {
        "operation": "add",
        "num1": 1,
        "num2": 2,
    }
    assert hl_file["tool"]["function"]["description"] == "Do arithmetic operations on two numbers."
    # TODO: pydantic is inconsistent by dumping either tuple or list
    assert calculator.json_schema == hl_file["tool"]["function"]

    Validator.check_schema(calculator.json_schema)


def test_union_type(opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter]):
    tracer, _ = opentelemetry_test_configuration

    @tool(opentelemetry_tracer=tracer)
    def foo(a: Union[int, float], b: float) -> float:
        return a + b

    assert foo.json_schema["parameters"]["properties"]["a"] == {
        "anyOf": [
            {"type": "integer"},
            {"type": "number"},
        ]
    }
    assert foo.json_schema["parameters"]["properties"]["b"] == {"type": "number"}
    assert foo.json_schema["parameters"]["required"] == ("a", "b")

    Validator.check_schema(foo.json_schema)


def test_not_required_parameter(
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    tracer, exporter = opentelemetry_test_configuration

    @tool(opentelemetry_tracer=tracer)
    def test_calculator(a: Optional[float], b: float) -> float:
        if a is None:
            a = 0
        return a + b

    assert test_calculator(3, 4) == 7
    assert len(exporter.get_finished_spans()) == 1
    assert test_calculator.json_schema["parameters"]["properties"]["a"] == {
        "type": ["number", "null"],
    }

    Validator.check_schema(test_calculator.json_schema)


def test_no_annotation_on_parameter(
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    # GIVEN an OTel configuration
    tracer, _ = opentelemetry_test_configuration

    # GIVEN a function annotated with @tool and without type hint on a parameter
    @tool(opentelemetry_tracer=tracer)
    def calculator(a: Optional[float], b) -> float:
        if a is None:
            a = 0
        return a + b

    # WHEN building the Tool kernel
    # THEN the JSON schema is correctly built and `b` is of `any` type
    # NOTE: JSONSchema dropped support for 'any' type, we include all types
    # as a workaround
    assert calculator.json_schema == {
        "description": "",
        "name": "calculator",
        "parameters": {
            "properties": {
                "a": {"type": ["number", "null"]},
                "b": {"type": ["string", "integer", "number", "boolean", "object", "array", "null"]},
            },
            "required": ("b",),
            "type": "object",
            "additionalProperties": False,
        },
        "strict": True,
    }

    Validator.check_schema(calculator.json_schema)


def test_dict_annotation_no_sub_types(
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    # GIVEN an OTel configuration
    tracer, _ = opentelemetry_test_configuration

    # GIVEN a function annotated with @tool and without type hint on a parameter
    @tool(opentelemetry_tracer=tracer)
    def calculator(a: Optional[float], b: dict) -> float:
        if a is None:
            a = 0
        return a + b["c"]

    # WHEN building the Tool kernel
    # THEN the JSON schema is correctly built and `b` accepts any type
    # on both keys and values
    assert calculator.json_schema == {
        "description": "",
        "name": "calculator",
        "parameters": {
            "properties": {
                "a": {"type": ["number", "null"]},
                "b": {
                    "type": "object",
                    "properties": {
                        "key": {"type": ["string", "integer", "number", "boolean", "object", "array", "null"]},
                        "value": {"type": ["string", "integer", "number", "boolean", "object", "array", "null"]},
                    },
                },
            },
            "required": ("b",),
            "type": "object",
            "additionalProperties": False,
        },
        "strict": True,
    }

    Validator.check_schema(calculator.json_schema)


def test_list_annotation_no_sub_types(
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    # GIVEN an OTel configuration
    tracer, _ = opentelemetry_test_configuration

    # GIVEN a function annotated with @tool and without type hint on a parameter
    @tool(opentelemetry_tracer=tracer)
    def calculator(a: Optional[float], b: Optional[list]) -> float:
        if a is None:
            a = 0
        sum = a
        if b is None:
            return sum
        for val in b:
            sum += val
        return sum

    # WHEN building the Tool kernel
    # THEN the JSON schema is correctly built and `b` accepts any type
    assert calculator.json_schema == {
        "description": "",
        "name": "calculator",
        "parameters": {
            "properties": {
                "a": {"type": ["number", "null"]},
                "b": {
                    "type": ["array", "null"],
                    "items": {"type": ["string", "integer", "number", "boolean", "object", "array", "null"]},
                },
            },
            "required": (),
            "type": "object",
            "additionalProperties": False,
        },
        "strict": True,
    }


def test_tuple_annotation_no_sub_types(
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    # GIVEN an OTel configuration
    tracer, _ = opentelemetry_test_configuration

    # GIVEN a function annotated with @tool and without type hint on a parameter
    @tool(opentelemetry_tracer=tracer)
    def calculator(a: Optional[float], b: Optional[tuple]) -> float:
        if a is None:
            a = 0
        sum = a
        if b is None:
            return sum
        for val in b:
            sum += val
        return sum

    # WHEN building the Tool kernel
    # THEN the JSON schema is correctly built and `b` accepts any type
    assert calculator.json_schema == {
        "description": "",
        "name": "calculator",
        "parameters": {
            "properties": {
                "a": {"type": ["number", "null"]},
                "b": {
                    "type": ["array", "null"],
                    "items": {"type": ["string", "integer", "number", "boolean", "object", "array", "null"]},
                },
            },
            "required": (),
            "type": "object",
            "additionalProperties": False,
        },
        "strict": True,
    }


def test_function_without_return_annotation(
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    # GIVEN an OTel configuration
    tracer, _ = opentelemetry_test_configuration

    # GIVEN a function annotated with @tool and without type hint on the return value
    # WHEN building the Tool kernel
    @tool(opentelemetry_tracer=tracer)
    def foo(a: Optional[float], b: float) -> float:
        """Add two numbers."""
        if a is None:
            a = 0
        return a + b

    # THEN the JSONSchema is valid
    Validator.check_schema(foo.json_schema)


def test_list_annotation_parameter(
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    # GIVEN an OTel configuration
    tracer, exporter = opentelemetry_test_configuration

    # WHEN defining a tool with a list parameter
    @tool(opentelemetry_tracer=tracer)
    def foo(to_join: list[str]) -> str:
        return " ".join(to_join)

    assert "a b c" == foo(to_join=["a", "b", "c"])

    # THEN the function call results in a Span
    assert len(exporter.get_finished_spans()) == 1
    # THEN the argument is correctly described in the JSON schema
    assert foo.json_schema["parameters"]["properties"]["to_join"] == {  # type: ignore
        "type": "array",
        "items": {"type": "string"},
    }
    # THEN the JSONSchema is valid
    Validator.check_schema(foo.json_schema)


def test_list_in_list_parameter_annotation(
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    # GIVEN an OTel configuration
    tracer, _ = opentelemetry_test_configuration

    # GIVEN a tool definition with a list of lists parameter
    # WHEN building the Tool Kernel
    @tool(opentelemetry_tracer=tracer)
    def nested_plain_join(to_join: list[list[str]]):
        return " ".join([val for sub_list in to_join for val in sub_list])

    # THEN the JSON schema is correctly built and parameter is correctly described
    assert nested_plain_join.json_schema["parameters"]["properties"]["to_join"] == {
        "type": "array",
        "items": {
            "type": "array",
            "items": {"type": "string"},
        },
    }

    # THEN the JSONSchema is valid
    Validator.check_schema(nested_plain_join.json_schema)


def test_complex_dict_annotation(
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    # GIVEN an OTel configuration
    tracer, _ = opentelemetry_test_configuration

    # GIVEN a tool definition with a dictionary parameter
    # WHEN building the Tool Kernel
    @tool(opentelemetry_tracer=tracer)
    def foo(a: dict[Union[int, str], list[str]]):
        return a

    # THEN the parameter is correctly described
    assert foo.json_schema["parameters"]["properties"]["a"] == {
        "type": "object",
        "properties": {
            "key": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
            "value": {"type": "array", "items": {"type": "string"}},
        },
    }

    # THEN the JSONSchema is valid
    Validator.check_schema(foo.json_schema)


def test_tuple_annotation(
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    # GIVEN an OTel configuration
    tracer, _ = opentelemetry_test_configuration

    # GIVEN a tool definition with a tuple parameter
    # WHEN building the Tool Kernel
    @tool(opentelemetry_tracer=tracer)
    def foo(a: Optional[tuple[int, Optional[str], float]]):
        return a

    # THEN the parameter is correctly described
    assert foo.json_schema["parameters"]["properties"]["a"] == {
        "type": ["array", "null"],
        "items": [
            {"type": "integer"},
            {"type": ["string", "null"]},
            {"type": "number"},
        ],
    }

    # THEN the JSONSchema is valid
    Validator.check_schema(foo.json_schema)


def test_tool_no_args(
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    # GIVEN an OTel configuration
    tracer, _ = opentelemetry_test_configuration

    # GIVEN a tool definition without arguments
    # WHEN building the Tool Kernel
    @tool(opentelemetry_tracer=tracer)
    def foo():
        return 42

    # THEN the JSON schema is correctly built
    assert foo.json_schema == {
        "description": "",
        "name": "foo",
        "parameters": {
            "properties": {},
            "required": [],
            "type": "object",
            "additionalProperties": False,
        },
        "strict": True,
    }

    # THEN the JSONSchema is valid
    Validator.check_schema(foo.json_schema)


def test_custom_types_throws(
    opentelemetry_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    # GIVEN an OTel configuration
    tracer, _ = opentelemetry_test_configuration

    # GIVEN a user-defined type
    class Foo(TypedDict):
        a: int  # type: ignore
        b: int  # type: ignore

    # WHEN defining a tool with a parameter of that type
    with pytest.raises(ValueError) as exc:

        @tool(opentelemetry_tracer=tracer)
        def foo_bar(foo: Foo):
            return foo.a + foo.b  # type: ignore

    # THEN a ValueError is raised
    assert exc.value.args[0].startswith("Error parsing signature of @tool annotated function foo_bar")


def test_tool_as_higher_order_function(
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    tracer, exporter = opentelemetry_hl_test_configuration

    def calculator(operation: str, num1: float, num2: float) -> float:
        """Do arithmetic operations on two numbers."""
        if operation == "add":
            return num1 + num2
        elif operation == "subtract":
            return num1 - num2
        elif operation == "multiply":
            return num1 * num2
        elif operation == "divide":
            return num1 / num2
        else:
            raise ValueError(f"Invalid operation: {operation}")

    higher_order_fn_tool = tool(opentelemetry_tracer=tracer)(calculator)

    @tool(opentelemetry_tracer=tracer)  # type: ignore
    def calculator(operation: str, num1: float, num2: float) -> float:
        """Do arithmetic operations on two numbers."""
        if operation == "add":
            return num1 + num2
        elif operation == "subtract":
            return num1 - num2
        elif operation == "multiply":
            return num1 * num2
        elif operation == "divide":
            return num1 / num2
        else:
            raise ValueError(f"Invalid operation: {operation}")

    higher_order_fn_tool(operation="add", num1=1, num2=2)
    calculator(operation="add", num1=1, num2=2)

    assert len(spans := exporter.get_finished_spans()) == 2

    hl_file_higher_order_fn = read_from_opentelemetry_span(
        span=spans[0],
        key=HUMANLOOP_FILE_KEY,
    )
    hl_file_decorated_fn = read_from_opentelemetry_span(
        span=spans[1],
        key=HUMANLOOP_FILE_KEY,
    )
    assert hl_file_higher_order_fn["tool"]["source_code"] == hl_file_decorated_fn["tool"]["source_code"]  # type: ignore
