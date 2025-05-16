import pytest

from humanloop import path_utils


@pytest.mark.parametrize(
    "input_path, expected_with_extension, expected_without_extension",
    [
        # Basic cases
        ("path/to/file.prompt", "path/to/file.prompt", "path/to/file"),
        ("path\\to\\file.agent", "path/to/file.agent", "path/to/file"),
        ("/leading/slashes/file.prompt", "leading/slashes/file.prompt", "leading/slashes/file"),
        ("trailing/slashes/file.agent/", "trailing/slashes/file.agent", "trailing/slashes/file"),
        ("multiple//slashes//file.prompt", "multiple/slashes/file.prompt", "multiple/slashes/file"),
        # Edge cases
        ("path/to/file with spaces.prompt", "path/to/file with spaces.prompt", "path/to/file with spaces"),
        (
            "path/to/file\\with\\backslashes.prompt",
            "path/to/file/with/backslashes.prompt",
            "path/to/file/with/backslashes",
        ),
        ("path/to/unicode/文件.prompt", "path/to/unicode/文件.prompt", "path/to/unicode/文件"),
        (
            "path/to/special/chars/!@#$%^&*().prompt",
            "path/to/special/chars/!@#$%^&*().prompt",
            "path/to/special/chars/!@#$%^&*()",
        ),
    ],
)
def test_normalize_path(input_path, expected_with_extension, expected_without_extension):
    """Test path normalization with various path formats."""
    # Test without stripping extension
    normalized = path_utils.normalize_path(input_path, strip_extension=False)
    assert normalized == expected_with_extension, f"Failed with strip_extension=False for '{input_path}'"

    # Test with extension stripping
    normalized = path_utils.normalize_path(input_path, strip_extension=True)
    assert normalized == expected_without_extension, f"Failed with strip_extension=True for '{input_path}'"
