import logging
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from humanloop.sync.sync_client import SyncClient
from humanloop.error import HumanloopRuntimeError


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
    assert sync_client.base_dir == tmp_path
    assert sync_client._cache_size == 10
    assert sync_client.SERIALIZABLE_FILE_TYPES == ["prompt", "agent"]


def test_normalize_path(sync_client: SyncClient):
    """Test path normalization functionality."""
    test_cases = [
        ("path/to/file.prompt", "path/to/file"),
        ("path\\to\\file.agent", "path/to/file"),
        ("/leading/slashes/file.prompt", "leading/slashes/file"),
        ("trailing/slashes/file.agent/", "trailing/slashes/file"),
        ("multiple//slashes//file.prompt", "multiple/slashes/file"),
    ]

    for input_path, expected in test_cases:
        assert sync_client._normalize_path(input_path) == expected


def test_is_file(sync_client: SyncClient):
    """Test file type detection."""
    assert sync_client.is_file("test.prompt")
    assert sync_client.is_file("test.agent")
    assert not sync_client.is_file("test.txt")
    assert not sync_client.is_file("test")


def test_save_and_read_file(sync_client: SyncClient):
    """Test saving and reading files."""
    content = "test content"
    path = "test/path"
    file_type = "prompt"

    # Test saving
    sync_client._save_serialized_file(content, path, "prompt")
    saved_path = sync_client.base_dir / path
    saved_path = saved_path.parent / f"{saved_path.stem}.{file_type}"
    assert saved_path.exists()

    # Test reading
    read_content = sync_client.get_file_content(path, file_type)
    assert read_content == content


def test_error_handling(sync_client: SyncClient):
    """Test error handling in various scenarios."""
    # Test file not found
    with pytest.raises(HumanloopRuntimeError, match="Local file not found"):
        sync_client.get_file_content("nonexistent", "prompt")

    # Test invalid file type
    with pytest.raises(ValueError, match="Unsupported file type"):
        sync_client._pull_file("test.txt")

    # Test API error
    with patch.object(sync_client.client.files, "retrieve_by_path", side_effect=Exception("API Error")):
        assert not sync_client._pull_file("test.prompt")


def test_cache_functionality(sync_client: SyncClient):
    """Test LRU cache functionality."""
    # Save a test file
    content = "test content"
    path = "test/path"
    file_type = "prompt"
    sync_client._save_serialized_file(content, path, file_type)  # type: ignore [arg-type] Ignore because we're deliberately testing an invalid literal

    # First read should hit disk
    sync_client.get_file_content(path, file_type)

    # Modify file on disk
    saved_path = sync_client.base_dir / f"{path}.{file_type}"
    saved_path.write_text("modified content")

    # Second read should use cache
    assert sync_client.get_file_content(path, file_type) == content

    # Clear cache and verify new content is read
    sync_client.clear_cache()
    assert sync_client.get_file_content(path, file_type) == "modified content"
