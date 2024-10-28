from typing import Any

from humanloop.decorators.tool import tool
from humanloop.otel.constants import HL_FILE_OT_KEY, HL_LOG_OT_KEY
from humanloop.otel.helpers import read_from_opentelemetry_span
from opentelemetry.sdk.trace import Tracer
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


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


def test_calculator_decorator(
    opentelemetry_hl_test_configuration: tuple[Tracer, InMemorySpanExporter],
):
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
    hl_file["tool"]["function"]["description"] == "Do arithmetic operations on two numbers."
    assert calculator.json_schema == hl_file["tool"]["function"]
