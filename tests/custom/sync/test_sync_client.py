import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from humanloop.sync.sync_client import SyncClient
from humanloop.error import HumanloopRuntimeError
from humanloop.types import FileType

@pytest.fixture
def mock_client():
    return Mock()

@pytest.fixture
def sync_client(mock_client, tmp_path):
    return SyncClient(
        client=mock_client,
        base_dir=str(tmp_path),
        cache_size=10,
        log_level=10  # DEBUG level for testing
    )

def test_init(sync_client, tmp_path):
    """Test basic initialization of SyncClient."""
    assert sync_client.base_dir == tmp_path
    assert sync_client._cache_size == 10
    assert sync_client.SERIALIZABLE_FILE_TYPES == ["prompt", "agent"]

def test_normalize_path(sync_client):
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

def test_is_file(sync_client):
    """Test file type detection."""
    assert sync_client.is_file("test.prompt")
    assert sync_client.is_file("test.agent")
    assert not sync_client.is_file("test.txt")
    assert not sync_client.is_file("test")

def test_save_and_read_file(sync_client):
    """Test saving and reading files."""
    content = "test content"
    path = "test/path"
    file_type = "prompt"
    
    # Test saving
    sync_client._save_serialized_file(content, path, file_type)
    saved_path = sync_client.base_dir / path
    saved_path = saved_path.parent / f"{saved_path.stem}.{file_type}"
    assert saved_path.exists()
    
    # Test reading
    read_content = sync_client.get_file_content(path, file_type)
    assert read_content == content

def test_pull_file(sync_client, mock_client):
    """Test pulling a single file."""
    mock_file = Mock()
    mock_file.type = "prompt"
    mock_file.path = "test/path"
    mock_file.raw_file_content = "test content"
    
    mock_client.files.retrieve_by_path.return_value = mock_file
    
    success = sync_client._pull_file("test/path.prompt")
    assert success
    assert mock_client.files.retrieve_by_path.called
    
    # Verify file was saved
    saved_path = sync_client.base_dir / "test/path.prompt"
    assert saved_path.exists()
    assert saved_path.read_text() == "test content"

def test_pull_directory(sync_client, mock_client):
    """Test pulling multiple files from a directory."""
    # Create mock responses for different pages
    def mock_list_files(*args, **kwargs):
        page = kwargs.get('page', 1)
        mock_response = Mock()
        
        if page == 1:
            mock_response.records = [
                Mock(
                    type="prompt",
                    path="test/path1",
                    raw_file_content="content1"
                ),
                Mock(
                    type="agent",
                    path="test/path2",
                    raw_file_content="content2"
                )
            ]
        else:
            # Return empty list for subsequent pages
            mock_response.records = []
            
        return mock_response
    
    # Set up the mock to use our function
    mock_client.files.list_files.side_effect = mock_list_files
    
    successful, failed = sync_client._pull_directory("test")
    assert len(successful) == 2
    assert len(failed) == 0
    
    # Verify files were saved
    assert (sync_client.base_dir / "test/path1.prompt").exists()
    assert (sync_client.base_dir / "test/path2.agent").exists()
    
    # Verify the mock was called with correct parameters
    mock_client.files.list_files.assert_any_call(
        type=["prompt", "agent"],
        page=1,
        include_raw_file_content=True,
        environment=None,
        path="test"
    )

def test_error_handling(sync_client):
    """Test error handling in various scenarios."""
    # Test file not found
    with pytest.raises(HumanloopRuntimeError, match="Local file not found"):
        sync_client.get_file_content("nonexistent", "prompt")
    
    # Test invalid file type
    with pytest.raises(ValueError, match="Unsupported file type"):
        sync_client._pull_file("test.txt")
    
    # Test API error
    with patch.object(sync_client.client.files, 'retrieve_by_path', side_effect=Exception("API Error")):
        assert not sync_client._pull_file("test.prompt")

def test_cache_functionality(sync_client):
    """Test LRU cache functionality."""
    # Save a test file
    content = "test content"
    path = "test/path"
    file_type = "prompt"
    sync_client._save_serialized_file(content, path, file_type)
    
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