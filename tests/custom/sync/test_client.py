import logging
from pathlib import Path
from typing import Literal
from unittest.mock import Mock, patch

import pytest

from humanloop.error import HumanloopRuntimeError
from humanloop.sync.sync_client import SerializableFileType, SyncClient


@pytest.fixture
def mock_client() -> Mock:
    return Mock()


@pytest.fixture
def sync_client(mock_client: Mock, tmp_path: Path) -> SyncClient:
    return SyncClient(
        client=mock_client,
        base_dir=str(tmp_path),
        cache_size=10,
        log_level=logging.DEBUG,  # DEBUG level for testing  # noqa: F821
    )


def test_init(sync_client: SyncClient, tmp_path: Path):
    """Test basic initialization of SyncClient."""
    # GIVEN a SyncClient instance
    # THEN it should be initialized with correct base directory, cache size and file types
    assert sync_client.base_dir == tmp_path
    assert sync_client._cache_size == 10
    assert sync_client.SERIALIZABLE_FILE_TYPES == frozenset(["prompt", "agent"])


def test_normalize_path(sync_client: SyncClient):
    """Test path normalization functionality."""
    # GIVEN various file paths with different formats
    test_cases = [
        # Input path, expected with strip_extension=False, expected with strip_extension=True
        ("path/to/file.prompt", "path/to/file.prompt", "path/to/file"),
        ("path\\to\\file.agent", "path/to/file.agent", "path/to/file"),
        ("trailing/slashes/file.agent/", "trailing/slashes/file.agent", "trailing/slashes/file"),
        ("multiple//slashes//file.prompt", "multiple/slashes/file.prompt", "multiple/slashes/file"),
    ]

    # Test with strip_extension=False (default)
    for input_path, expected_with_ext, _ in test_cases:
        # WHEN they are normalized without stripping extension
        normalized = sync_client._normalize_path(input_path, strip_extension=False)
        # THEN they should be converted to the expected format with extension
        assert normalized == expected_with_ext

    # Test with strip_extension=True
    for input_path, _, expected_without_ext in test_cases:
        # WHEN they are normalized with extension stripping
        normalized = sync_client._normalize_path(input_path, strip_extension=True)
        # THEN they should be converted to the expected format without extension
        assert normalized == expected_without_ext

    # Test absolute path raises error
    with pytest.raises(HumanloopRuntimeError, match="Absolute paths are not supported"):
        sync_client._normalize_path("/leading/slashes/file.prompt")


def test_is_file(sync_client: SyncClient):
    """Test file type detection."""
    # GIVEN various file paths
    # WHEN checking if they are valid file types
    # THEN only .prompt and .agent files should return True
    assert sync_client.is_file("test.prompt")
    assert sync_client.is_file("test.agent")
    assert not sync_client.is_file("test.txt")
    assert not sync_client.is_file("test")


def test_save_and_read_file(sync_client: SyncClient):
    """Test saving and reading files."""
    # GIVEN a file content and path
    content = "test content"
    path = "test/path"
    file_type: SerializableFileType = "prompt"

    # WHEN saving the file
    sync_client._save_serialized_file(content, path, "prompt")
    saved_path = sync_client.base_dir / path
    saved_path = saved_path.parent / f"{saved_path.stem}.{file_type}"

    # THEN the file should exist on disk
    assert saved_path.exists()

    # WHEN reading the file
    read_content = sync_client.get_file_content(path, file_type)

    # THEN the content should match
    assert read_content == content


def test_error_handling(sync_client: SyncClient):
    """Test error handling in various scenarios."""
    # GIVEN a nonexistent file
    # WHEN trying to read it
    # THEN a HumanloopRuntimeError should be raised
    with pytest.raises(HumanloopRuntimeError, match="Local file not found"):
        sync_client.get_file_content("nonexistent", "prompt")

    # GIVEN an API error
    # WHEN trying to pull a file
    # THEN it should return False
    with patch.object(sync_client.client.files, "retrieve_by_path", side_effect=Exception("API Error")):
        assert not sync_client._pull_file("test.prompt")


def test_cache_functionality(sync_client: SyncClient):
    """Test LRU cache functionality."""
    # GIVEN a test file
    content = "test content"
    path = "test/path"
    file_type: Literal["prompt", "agent"] = "prompt"
    sync_client._save_serialized_file(content, path, file_type)

    # WHEN reading the file for the first time
    sync_client.get_file_content(path, file_type)
    # THEN it should hit disk (implicitly verified by no cache hit)

    # WHEN modifying the file on disk
    saved_path = sync_client.base_dir / f"{path}.{file_type}"
    saved_path.write_text("modified content")

    # THEN subsequent reads should use cache
    assert sync_client.get_file_content(path, file_type) == content

    # WHEN clearing the cache
    sync_client.clear_cache()

    # THEN new content should be read from disk
    assert sync_client.get_file_content(path, file_type) == "modified content"
