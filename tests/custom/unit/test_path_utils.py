from humanloop import path_utils


def test_normalize_path():
    """Test path normalization functionality."""
    # GIVEN various file paths with different formats
    test_cases = [
        # Input path, expected with strip_extension=False, expected with strip_extension=True
        ("path/to/file.prompt", "path/to/file.prompt", "path/to/file"),
        ("path\\to\\file.agent", "path/to/file.agent", "path/to/file"),
        ("/leading/slashes/file.prompt", "leading/slashes/file.prompt", "leading/slashes/file"),
        ("trailing/slashes/file.agent/", "trailing/slashes/file.agent", "trailing/slashes/file"),
        ("multiple//slashes//file.prompt", "multiple/slashes/file.prompt", "multiple/slashes/file"),
    ]

    # Test with strip_extension=False (default)
    for input_path, expected_with_ext, _ in test_cases:
        # WHEN they are normalized without stripping extension
        normalized = path_utils.normalize_path(input_path, strip_extension=False)
        # THEN they should be converted to the expected format with extension
        assert normalized == expected_with_ext

    # Test with strip_extension=True
    for input_path, _, expected_without_ext in test_cases:
        # WHEN they are normalized with extension stripping
        normalized = path_utils.normalize_path(input_path, strip_extension=True)
        # THEN they should be converted to the expected format without extension
        assert normalized == expected_without_ext
