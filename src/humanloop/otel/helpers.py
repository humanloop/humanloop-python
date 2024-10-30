from typing import Union

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanKind
from opentelemetry.util.types import AttributeValue

from humanloop.otel.constants import HL_FILE_OT_KEY, HL_LOG_OT_KEY

NestedDict = dict[str, Union["NestedDict", AttributeValue]]
NestedList = list[Union["NestedList", NestedDict]]


def _list_to_ott(lst: NestedList) -> NestedDict:
    """Transforms list of values to be written into a dictionary with index values as keys.

    When writing to Otel span attributes, only primitive values or lists are allowed.
    Nested dictionaries must be linearised. For example, writing to span attribute `foo`
    the dictionary value {'a': 7, 'b': 'hello'} would translated in the span attributes
    dictionary to look like:
    ```python
    {
        'foo.a': 7,
        'foo.b': 'hello'
    }
    ```

    Calling :func:`write_to_opentelemetry_span` with a list for write value will have
    the list transformed into a pseudo-dictionary with index values as keys.

    Examples:
    ```python
    _list_to_ott([1, 2, 'a']) == {'0': 1, '1': 2, '2': 'a'}
    _list_to_ott([
        "baz",
        {'a': 6, 'b': 'hello'}
    ]) == {
        '0': 'baz',
        '1.a': 6,
        '1.b': 'hello'
    }
    ```
    """
    return {str(idx): val if not isinstance(val, list) else _list_to_ott(val) for idx, val in enumerate(lst)}


def write_to_opentelemetry_span(
    span: ReadableSpan,
    value: Union[NestedDict, NestedList],
    key: str = "",
) -> None:
    """Write a Python object to the OpenTelemetry Span's attributes. Reverse of :func:`read_from_opentelemetry_span`.

    Note: OTel will complain about falsy values other then None, and keys with value set
    to None will be silently dropped. Consider adding a placeholder value if the key should
    be present in the span attributes.

    :param span: OpenTelemetry span to write values to

    :param value: Python object to write to the span attributes. Can also be a primitive value.

    :param key: Key prefix to write to the span attributes. The path to the values does not need to exist in the span attributes.
    """
    to_write_copy: Union[dict, AttributeValue]
    if isinstance(value, list):
        to_write_copy = _list_to_ott(value)
    else:
        to_write_copy = dict(value)
    linearised_attributes: dict[str, AttributeValue] = {}
    work_stack: list[tuple[str, Union[AttributeValue, NestedDict]]] = [(key, to_write_copy)]
    """
    Recurse through the dictionary value, building the OTel format keys in a DFS manner.

    Example:
    ```python
    {
        'foo': {
            'a': 7,
            'b': 'hello'
        },
        "baz": [42, 43]
    }

    1. Visit foo, push ('foo.a', 7), ('foo.b', 'hello') to stack
    2. Visit baz, push ('baz.0', 42), ('baz.1', 43) to stack
    3. Take each primitive key-value from the stack and write to the span attributes,
        resulting in:
        {
            'foo.a': 7,
            'foo.b': 'hello',
            'baz.0': 42,
            'baz.1': 43
        }
    ```
    """
    while len(work_stack) > 0:
        key, value = work_stack.pop()  # type: ignore
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                work_stack.append((f"{key}.{sub_key}" if key else sub_key, sub_value))
        else:
            linearised_attributes[key] = value  # type: ignore
    for final_key, final_value in linearised_attributes.items():
        span._attributes[final_key] = final_value  # type: ignore


def read_from_opentelemetry_span(span: ReadableSpan, key: str = "") -> NestedDict:
    """Read a value from the OpenTelemetry span attributes.

    OpenTelemetry liniarises dictionaries and lists, storing only primitive values
    in the span attributes. This function reconstructs the original structure from
    a key prefix.

    :param span: OpenTelemetry span to read values from
    :param key: Key prefix to read from the span attributes


    Examples:
    `span.attributes` contains the following attributes:
    ```python
        foo.x.y = 7
        foo.x.z.a = 'hello'
        foo.x.z.b = 'world'
        baz.0 = 42
        baz.1 = 43
    ```

    `read_from_opentelemetry_span(span, key='foo')` returns:
    ```python
    {
        'x': {
            'y': 7,
            'z': {
                'a': 'hello',
                'b': 'world'
            }
        }
    }
    ```

    `read_from_opentelemetry_span(span, key='foo.x')` returns:
    ```python
    {
        'y': 7,
        'z': {
            'a': 'hello',
            'b': 'world'
        }
    }
    ```

    `read_from_opentelemetry_span(span, key='baz')` returns:
    ```python
    [42, 43]
    ```
    """
    if span._attributes is None:
        raise ValueError("Span attributes are empty")

    result: dict[str, Union[dict, AttributeValue]] = {}

    to_process: list[tuple[str, Union[dict, AttributeValue]]] = []
    for span_key, span_value in span._attributes.items():  # type: ignore
        if key == "":
            # No key prefix, add to root
            to_process.append((f"{key}.{span_key}", span_value))
        elif span_key.startswith(key):
            # Remove the key prefix and the first dot
            to_process.append((span_key, span_value))

    if not to_process:
        if key == "":
            # Empty span attributes
            return result
        raise KeyError(f"Key {key} not found in span attributes")

    for span_key, span_value in to_process:  # type: ignore
        parts = span_key.split(".")
        len_parts = len(parts)
        sub_result: dict[str, Union[dict, AttributeValue]] = result
        for idx, part in enumerate(parts):
            if idx == len_parts - 1:
                sub_result[part] = span_value
            else:
                if part not in sub_result:
                    sub_result[part] = {}
                sub_result = sub_result[part]  # type: ignore

    for part in key.split("."):
        result = result[part]  # type: ignore

    return result


def is_llm_provider_call(span: ReadableSpan) -> bool:
    """Determines if the span was created by an Instrumentor for LLM provider clients."""
    if not span.instrumentation_scope:
        return False
    span_instrumentor_name = span.instrumentation_scope.name
    # Match against the prefix of the Instrumentor name since
    # the name might be version dependent e.g.
    # "opentelemetry.instrumentation.openai.v1"
    return span.kind == SpanKind.CLIENT and any(
        span_instrumentor_name.startswith(instrumentor)
        for instrumentor in [
            "opentelemetry.instrumentation.openai",
            "opentelemetry.instrumentation.groq",
            "opentelemetry.instrumentation.anthropic",
            "opentelemetry.instrumentation.cohere",
            "opentelemetry.instrumentation.replicate",
        ]
    )


def is_humanloop_span(span: ReadableSpan) -> bool:
    """Check if the Span was created by the Humanloop SDK."""
    try:
        # Valid spans will have keys with the HL_FILE_OT_KEY and HL_LOG_OT_KEY prefixes present
        read_from_opentelemetry_span(span, key=HL_FILE_OT_KEY)
        read_from_opentelemetry_span(span, key=HL_LOG_OT_KEY)
    except KeyError:
        return False
    return True


def module_is_installed(module_name: str) -> bool:
    """Returns true if the current Python environment has the module installed.

    Used to check if a library that is instrumentable exists in the current environment.
    """
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True