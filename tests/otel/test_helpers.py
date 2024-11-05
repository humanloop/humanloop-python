import pytest
from humanloop.otel.helpers import read_from_opentelemetry_span, write_to_opentelemetry_span
from opentelemetry.sdk.trace import Span


def test_read_empty(test_span: Span):
    assert read_from_opentelemetry_span(test_span) == {}


def test_read_non_existent_key(test_span: Span):
    with pytest.raises(KeyError):
        assert read_from_opentelemetry_span(test_span, "key") == {}
    write_to_opentelemetry_span(test_span, {"x": 7, "y": "foo"}, key="key")
    # NOTE: attributes cannot be None at this point
    assert dict(test_span.attributes) == {  # type: ignore
        "key.x": 7,
        "key.y": "foo",
    }
    with pytest.raises(KeyError):
        assert read_from_opentelemetry_span(test_span, "key.z") is None


def test_simple_dict(test_span: Span):
    write_to_opentelemetry_span(test_span, {"x": 7, "y": "foo"}, "key")
    # NOTE: attributes cannot be None at this point
    assert dict(test_span.attributes) == {  # type: ignore
        "key.x": 7,
        "key.y": "foo",
    }
    assert read_from_opentelemetry_span(test_span, "key") == {"x": 7, "y": "foo"}


def test_no_prefix(test_span: Span):
    write_to_opentelemetry_span(test_span, {"x": 7, "y": "foo"})
    # NOTE: attributes cannot be None at this point
    assert dict(test_span.attributes) == {  # type: ignore
        "x": 7,
        "y": "foo",
    }
    assert read_from_opentelemetry_span(test_span) == {"x": 7, "y": "foo"}


def test_nested_object(test_span: Span):
    write_to_opentelemetry_span(test_span, {"x": 7, "y": {"z": "foo"}}, "key")
    # NOTE: attributes cannot be None at this point
    assert dict(test_span.attributes) == {  # type: ignore
        "key.x": 7,
        "key.y.z": "foo",
    }
    assert read_from_opentelemetry_span(test_span, "key") == {"x": 7, "y": {"z": "foo"}}


def test_list(test_span: Span):
    write_to_opentelemetry_span(
        test_span,
        [{"x": 7, "y": "foo"}, {"z": "bar"}],  # type: ignore
        "key",
    )  # type: ignore
    # NOTE: attributes cannot be None at this point
    assert dict(test_span.attributes) == {  # type: ignore
        "key.0.x": 7,
        "key.0.y": "foo",
        "key.1.z": "bar",
    }
    assert read_from_opentelemetry_span(test_span, "key") == {
        "0": {"x": 7, "y": "foo"},
        "1": {"z": "bar"},
    }


def test_list_no_prefix(test_span: Span):
    write_to_opentelemetry_span(
        test_span,
        [{"x": 7, "y": "foo"}, {"z": "bar"}],  # type: ignore
    )
    # NOTE: attributes cannot be None at this point
    assert dict(test_span.attributes) == {  # type: ignore
        "0.x": 7,
        "0.y": "foo",
        "1.z": "bar",
    }
    assert read_from_opentelemetry_span(test_span) == {
        "0": {"x": 7, "y": "foo"},
        "1": {"z": "bar"},
    }


def test_multiple_nestings(test_span: Span):
    write_to_opentelemetry_span(
        test_span,
        [
            {"x": 7, "y": "foo"},
            [{"z": "bar"}, {"a": 42}],
        ],  # type: ignore
        "key",
    )
    # NOTE: attributes cannot be None at this point
    assert dict(test_span.attributes) == {  # type: ignore
        "key.0.x": 7,
        "key.0.y": "foo",
        "key.1.0.z": "bar",
        "key.1.1.a": 42,
    }
    assert read_from_opentelemetry_span(test_span, "key") == {
        "0": {"x": 7, "y": "foo"},
        "1": {
            "0": {"z": "bar"},
            "1": {"a": 42},
        },
    }


def test_read_mixed_numeric_string_keys(test_span: Span):
    test_span.set_attributes(
        {
            "key.0.x": 7,
            "key.0.y": "foo",
            "key.a.z": "bar",
            "key.a.a": 42,
        }
    )
    assert read_from_opentelemetry_span(span=test_span, key="key") == {  # type: ignore
        "0": {"x": 7, "y": "foo"},
        "a": {"z": "bar", "a": 42},
    }
    assert read_from_opentelemetry_span(span=test_span) == {  # type: ignore
        "key": {
            "0": {"x": 7, "y": "foo"},
            "a": {"z": "bar", "a": 42},
        }
    }


def test_sub_key_same_as_key(test_span: Span):
    write_to_opentelemetry_span(test_span, {"key": 7}, "key")
    # NOTE: attributes cannot be None at this point
    assert dict(test_span.attributes) == {  # type: ignore
        "key.key": 7,
    }
    assert read_from_opentelemetry_span(test_span, "key") == {"key": 7}


def test_read_nested_key(test_span: Span):
    test_span.set_attributes({"key.x": 7, "key.y.z": "foo"})
    assert read_from_opentelemetry_span(span=test_span, key="key.y") == {"z": "foo"}


def test_write_read_sub_key(test_span: Span):
    write_to_opentelemetry_span(test_span, {"x": 7, "y": "foo"}, "key")
    assert read_from_opentelemetry_span(test_span, "key.x") == 7
    assert read_from_opentelemetry_span(test_span, "key.y") == "foo"
    assert read_from_opentelemetry_span(test_span, "key") == {"x": 7, "y": "foo"}


def test_write_drops_dict_all_null_values(test_span: Span):
    # GIVEN a test_span to which a value with null values is written
    # NOTE: mypy complains about None value in the dict, but it is intentionally under test
    write_to_opentelemetry_span(test_span, {"x": None, "y": None}, "key")  # type: ignore
    # WHEN reading the value from the span
    # THEN the value is not present in the span attributes
    assert "key" not in test_span.attributes  # type: ignore
    with pytest.raises(KeyError):
        assert read_from_opentelemetry_span(test_span, "key") == {}


def test_write_drops_null_value_from_dict(test_span: Span):
    # GIVEN a test_span to which a dict with some null values are written
    # NOTE: mypy complains about None value in the dict, but it is intentionally under test
    write_to_opentelemetry_span(test_span, {"x": 2, "y": None}, "key")  # type: ignore
    # WHEN reading the values from the span
    # THEN the value with null value is not present in the span attributes
    assert read_from_opentelemetry_span(test_span, "key") == {"x": 2}
