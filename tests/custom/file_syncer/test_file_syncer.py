import logging
from pathlib import Path
from typing import Literal
from unittest.mock import Mock, patch

import pytest

from humanloop.error import HumanloopRuntimeError
from humanloop.sync.file_syncer import FileSyncer, SerializableFileType


@pytest.fixture
def mock_client() -> Mock:
    return Mock()


@pytest.fixture
def file_syncer(mock_client: Mock, tmp_path: Path) -> FileSyncer:
    return FileSyncer(
        client=mock_client,
        base_dir=str(tmp_path),
        cache_size=10,
        log_level=logging.DEBUG,  # DEBUG level for testing  # noqa: F821
    )


def test_init(file_syncer: FileSyncer, tmp_path: Path):
    """Test basic initialization of FileSyncer."""
    # GIVEN a FileSyncer instance
    # THEN it should be initialized with correct base directory, cache size and file types
    assert file_syncer.base_dir == tmp_path  # Compare Path objects directly
    assert file_syncer._cache_size == 10
    assert file_syncer.SERIALIZABLE_FILE_TYPES == frozenset(["prompt", "agent"])


def test_is_file(file_syncer: FileSyncer):
    """Test file type detection with case insensitivity."""
    # GIVEN a FileSyncer instance

    # WHEN checking various file paths with different extensions and cases
    # THEN .prompt and .agent files (of any case) should return True

    # Standard lowercase extensions
    assert file_syncer.is_file("test.prompt")
    assert file_syncer.is_file("test.agent")

    # Uppercase extensions (case insensitivity)
    assert file_syncer.is_file("test.PROMPT")
    assert file_syncer.is_file("test.AGENT")
    assert file_syncer.is_file("test.Prompt")
    assert file_syncer.is_file("test.Agent")

    # With whitespace
    assert file_syncer.is_file(" test.prompt ")
    assert file_syncer.is_file(" test.agent ")

    # WHEN checking paths with invalid or no extensions
    # THEN they should return False

    # Invalid file types
    assert not file_syncer.is_file("test.txt")
    assert not file_syncer.is_file("test.json")
    assert not file_syncer.is_file("test.py")

    # No extension
    assert not file_syncer.is_file("test")
    assert not file_syncer.is_file("prompt")
    assert not file_syncer.is_file("agent")

    # Partial extensions
    assert not file_syncer.is_file("test.prom")
    assert not file_syncer.is_file("test.age")


def test_save_and_read_file(file_syncer: FileSyncer):
    """Test saving and reading files."""
    # GIVEN a file content and path
    content = "test content"
    path = "test/path"
    file_type: SerializableFileType = "prompt"

    # WHEN saving the file
    file_syncer._save_serialized_file(content, path, "prompt")
    saved_path = file_syncer.base_dir / path
    saved_path = saved_path.parent / f"{saved_path.stem}.{file_type}"

    # THEN the file should exist on disk
    assert saved_path.exists()

    # WHEN reading the file
    read_content = file_syncer.get_file_content(path, file_type)

    # THEN the content should match
    assert read_content == content


def test_error_handling(file_syncer: FileSyncer):
    """Test error handling in various scenarios."""
    # GIVEN a nonexistent file
    # WHEN trying to read it
    # THEN a HumanloopRuntimeError should be raised
    with pytest.raises(HumanloopRuntimeError, match="Local file not found"):
        file_syncer.get_file_content("nonexistent", "prompt")

    # GIVEN an API error
    # WHEN trying to pull a file
    # THEN it should return False
    with patch.object(file_syncer.client.files, "retrieve_by_path", side_effect=Exception("API Error")):
        assert not file_syncer._pull_file("test.prompt")


def test_cache_functionality(file_syncer: FileSyncer):
    """Test LRU cache functionality."""
    # GIVEN a test file
    content = "test content"
    path = "test/path"
    file_type: Literal["prompt", "agent"] = "prompt"
    file_syncer._save_serialized_file(content, path, file_type)

    # WHEN reading the file for the first time
    file_syncer.get_file_content(path, file_type)
    # THEN it should hit disk (implicitly verified by no cache hit)

    # WHEN modifying the file on disk
    saved_path = file_syncer.base_dir / f"{path}.{file_type}"
    saved_path.write_text("modified content")

    # THEN subsequent reads should use cache
    assert file_syncer.get_file_content(path, file_type) == content

    # WHEN clearing the cache
    file_syncer.clear_cache()

    # THEN new content should be read from disk
    assert file_syncer.get_file_content(path, file_type) == "modified content"
