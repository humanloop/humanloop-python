from typing import Any, Optional, TypedDict, Union

import pytest
from humanloop.decorators.tool import tool
from humanloop.otel.constants import HL_FILE_OT_KEY, HL_LOG_OT_KEY
from humanloop.otel.helpers import read_from_opentelemetry_span
from humanloop.types.tool_kernel_request import ToolKernelRequest
from jsonschema.protocols import Validator
from opentelemetry.sdk.trace import Tracer
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_calculator_decorator(
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    @tool()
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

    # GIVEN a test OpenTelemetry configuration
    _, exporter = opentelemetry_hl_test_configuration
    # WHEN calling the @tool decorated function
    result = calculator(operation="add", num1=1, num2=2)
    # THEN a single span is created and the log and file attributes are correctly set
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    hl_file: dict[str, Any] = read_from_opentelemetry_span(span=spans[0], key=HL_FILE_OT_KEY)
    hl_log: dict[str, Any] = read_from_opentelemetry_span(span=spans[0], key=HL_LOG_OT_KEY)
    assert hl_log["output"] == result == 3
    assert hl_log["inputs"] == {
        "operation": "add",
        "num1": 1,
        "num2": 2,
    }
    assert hl_file["tool"]["function"]["description"] == "Do arithmetic operations on two numbers."
    # TODO: pydantic is inconsistent by dumping either tuple or list
    assert calculator.json_schema == hl_file["tool"]["function"]

    Validator.check_schema(calculator.json_schema)


def test_union_type():
    @tool()
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
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    _, exporter = opentelemetry_hl_test_configuration

    @tool()
    def test_calculator(a: Optional[float], b: float) -> float:
        if a is None:
            a = 0
        return a + b

    assert test_calculator(3, 4) == 7
    assert len(spans := exporter.get_finished_spans()) == 1
    tool_kernel = ToolKernelRequest.model_validate(read_from_opentelemetry_span(spans[0], HL_FILE_OT_KEY)["tool"])
    assert test_calculator.json_schema["parameters"]["properties"]["a"] == {"type": ["number", "null"]}
    assert tool_kernel.function.parameters["required"] == ("b",)  # type: ignore

    Validator.check_schema(test_calculator.json_schema)


def test_no_annotation_on_parameter():
    # GIVEN a function annotated with @tool and without type hint on a parameter
    @tool()
    def calculator(a: Optional[float], b) -> float:
        if a is None:
            a = 0
        return a + b

    # WHEN building the Tool kernel
    # THEN the JSON schema is correctly built and `b` is is of `any` type
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
        },
        "strict": True,
    }

    Validator.check_schema(calculator.json_schema)


def test_dict_annotation_no_sub_types():
    # GIVEN a function annotated with @tool and without type hint on a parameter
    @tool()
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
        },
        "strict": True,
    }

    Validator.check_schema(calculator.json_schema)


def test_list_annotation_no_sub_types():
    # GIVEN a function annotated with @tool and without type hint on a parameter
    @tool()
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
        },
        "strict": True,
    }


def test_tuple_annotation_no_sub_types():
    # GIVEN a function annotated with @tool and without type hint on a parameter
    @tool()
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
        },
        "strict": True,
    }


def test_function_without_return_annotation():
    # GIVEN a function annotated with @tool and without type hint on the return value
    # WHEN building the Tool kernel
    @tool()
    def foo(a: Optional[float], b: float) -> float:
        """Add two numbers."""
        if a is None:
            a = 0
        return a + b

    # THEN the JSONSchema is valid
    Validator.check_schema(foo.json_schema)


def test_list_annotation_parameter(
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
    # GIVEN an OTel configuration with HL Processor
    _, exporter = opentelemetry_hl_test_configuration

    # WHEN defining a tool with a list parameter
    @tool()
    def foo(to_join: list[str]) -> str:
        return " ".join(to_join)

    assert "a b c" == foo(to_join=["a", "b", "c"])

    # THEN the function call results in a Span
    assert len(spans := exporter.get_finished_spans()) == 1
    # THEN a valid Tool Kernel can be parsed from the Span
    tool_kernel = ToolKernelRequest.model_validate(
        read_from_opentelemetry_span(
            spans[0],
            HL_FILE_OT_KEY,
        )["tool"]
    )
    # THEN the argument is present in the Tool Kernel
    assert "to_join" in tool_kernel.function.parameters["required"]  # type: ignore
    # THEN the argument is correctly described in the JSON schema
    assert tool_kernel.function.parameters["properties"]["to_join"] == {  # type: ignore
        "type": "array",
        "items": {"type": "string"},
    }
    # THEN the JSONSchema is valid
    Validator.check_schema(foo.json_schema)


def test_list_in_list_parameter_annotation():
    # GIVEN a tool definition with a list of lists parameter
    # WHEN building the Tool Kernel
    @tool()
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


def test_complex_dict_annotation():
    # GIVEN a tool definition with a dictionary parameter
    # WHEN building the Tool Kernel
    @tool()
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


def test_tuple_annotation():
    # GIVEN a tool definition with a tuple parameter
    # WHEN building the Tool Kernel
    @tool()
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


def test_strict_false():
    # GIVEN a tool definition with strict=False
    # WHEN building the Tool Kernel
    @tool(strict=False)
    def foo(a: int, b: int) -> int:
        return a + b

    # THEN the JSON schema is correctly built
    assert foo.json_schema["strict"] is False

    # THEN the JSONSchema is valid
    Validator.check_schema(foo.json_schema)


def test_tool_no_args():
    # GIVEN a tool definition without arguments
    # WHEN building the Tool Kernel
    @tool()
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
        },
        "strict": True,
    }

    # THEN the JSONSchema is valid
    Validator.check_schema(foo.json_schema)


def test_custom_types_throws():
    # GIVEN a user-defined type
    class Foo(TypedDict):
        a: int  # type: ignore
        b: int  # type: ignore

    # WHEN defining a tool with a parameter of that type
    with pytest.raises(ValueError) as exc:

        @tool()
        def foo_bar(foo: Foo):
            return foo.a + foo.b  # type: ignore

    # THEN a ValueError is raised
    assert exc.value.args[0].startswith("foo_bar: Unsupported type hint")
