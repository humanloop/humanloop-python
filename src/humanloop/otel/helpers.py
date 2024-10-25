import builtins
from typing import Any

from opentelemetry.sdk.trace import Span

from humanloop.otel.constants import HL_FILE_OT_KEY, HL_LOG_OT_KEY


def write_to_opentelemetry_span(span: Span, value: Any, key: str = "") -> None:
    """Reverse of read_from_opentelemetry_span. Writes a Python object to the OpenTelemetry Span's attributes.

    See `read_from_opentelemetry_span` for more information.

    Arguments:
        span: OpenTelemetry span to write values to
        value: Python object to write to the span attributes. Can also be a primitive value.
        key: Key prefix to write to the span attributes. The path to the values does not
            need to exist in the span attributes.
    """
    to_write = dict()
    _linear_object(to_write, value)
    for k, v in to_write.items():
        # OTT
        if v is not None:
            span._attributes[f"{key}.{k}" if key != "" else k] = v
    # with _cache_lock:
    #     _cache[(span.context.span_id, key)] = value


def read_from_opentelemetry_span(span: Span, key: str = "") -> dict | list:
    """Read a value from the OpenTelemetry span attributes.

    OpenTelemetry liniarises dictionaries and lists, storing only primitive values
    in the span attributes. This function reconstructs the original structure from
    a key prefix.

    Arguments:
        span: OpenTelemetry span to read values from
        key: Key prefix to read from the span attributes

    Returns:
        Python object stored in the span attributes under the key prefix.

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

    result = dict()

    to_process: list[tuple[str, Any]] = []
    for span_key, span_value in span._attributes.items():
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

    for span_key, span_value in to_process:
        parts = span_key.split(".")
        len_parts = len(parts)
        sub_result = result
        for idx, part in enumerate(parts):
            if idx == len_parts - 1:
                sub_result[part] = span_value
            else:
                if part not in sub_result:
                    sub_result[part] = dict()
                sub_result = sub_result[part]

    result = _dict_to_list(result)
    for part in key.split("."):
        result = result[part]
    return result


def _linear_object(obj: dict, current: dict | list | Any, key: str = ""):
    """Linearise a Python object into a dictionary.

    Method recurses on the `current` argument, collecting all primitive values and their
    path in the objects, then storing them in the `obj` dictionary in the end.

    Arguments:
        obj: Dictionary to store the linearised object
        current: Python object to linearise. Used in recursivity when a complex
            value is encountered.
        key: Key prefix to store the values in the `obj` dictionary. Keys are added
            incrementally as the function recurses.

    Examples:
    ```python
    result = dict()
    _linear_object(result, {'a': 1, 'b': {'c': 2, d: [4, 5]}})

    # result is now:
    {
        'a': 1,
        'b.c': 2,
        'b.d.0': 4,
        'b.d.1': 5
    }
    ```

    """
    match type(current):
        case builtins.dict:
            for k, v in current.items():
                _linear_object(obj, v, f"{key}.{k}" if key != "" else k)
        case builtins.list:
            for idx, v in enumerate(current):
                _linear_object(obj, v, f"{key}.{idx}" if key != "" else str(idx))
        case _:
            obj[key] = current


def _dict_to_list(d: dict[str, Any]) -> dict | list:
    """Interpret number keys parsed by the read_from_opentelemetry_span function as lists.

    read_from_opentelemetry_span assumes all sub-keys in a path such as foo.0.bar are keys in
    dictionaries. This method revisits the final result, and transforms the keys in lists where
    appropriate.
    """
    is_list = all(key.isdigit() for key in d.keys())
    if is_list:
        return [_dict_to_list(val) if isinstance(val, dict) else val for val in d.values()]
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = _dict_to_list(value)
    return d


def is_llm_provider_call(span: Span) -> bool:
    """Determines if the span was created by an Instrumentor for LLM provider clients."""
    return "llm.request.type" in span.attributes


def is_humanloop_span(span: Span) -> bool:
    """Determines if the span was created by the Humanloop SDK."""
    try:
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
