import pytest
from humanloop import Humanloop, FileType
from pathlib import Path
from typing import List, NamedTuple


class SyncableFile(NamedTuple):
    path: str
    type: FileType
    model: str
    id: str = ""


@pytest.fixture
def test_file_structure(humanloop_client: Humanloop, get_test_path) -> List[SyncableFile]:
    """Creates a predefined structure of files in Humanloop for testing sync"""
    files: List[SyncableFile] = [
        SyncableFile(
            path="prompts/gpt-4",
            type="prompt",
            model="gpt-4",
        ),
        SyncableFile(
            path="prompts/gpt-4o",
            type="prompt",
            model="gpt-4o",
        ),
        SyncableFile(
            path="prompts/nested/complex/gpt-4o",
            type="prompt",
            model="gpt-4o",
        ),
        SyncableFile(
            path="agents/gpt-4",
            type="agent",
            model="gpt-4",
        ),
        SyncableFile(
            path="agents/gpt-4o",
            type="agent",
            model="gpt-4o",
        ),
    ]

    # Create the files in Humanloop
    created_files = []
    for file in files:
        full_path = get_test_path(file.path)
        if file.type == "prompt":
            response = humanloop_client.prompts.upsert(
                path=full_path,
                model=file.model,
            )
        elif file.type == "agent":
            response = humanloop_client.agents.upsert(
                path=full_path,
                model=file.model,
            )
        created_files.append(SyncableFile(path=full_path, type=file.type, model=file.model, id=response.id))

    return created_files


@pytest.fixture
def cleanup_local_files():
    """Cleanup any locally synced files after tests"""
    yield
    # Clean up the local humanloop directory after tests
    local_dir = Path("humanloop")
    if local_dir.exists():
        import shutil

        shutil.rmtree(local_dir)


def test_sync_basic(humanloop_client: Humanloop, test_file_structure: List[SyncableFile], cleanup_local_files):
    """Test that humanloop.sync() correctly syncs remote files to local filesystem"""
    # Run the sync
    successful_files = humanloop_client.sync()

    # Verify each file was synced correctly
    for file in test_file_structure:
        # Get the extension based on file type: .prompt, .agent
        extension = f".{file.type}"

        # The local path should mirror the remote path structure
        local_path = Path("humanloop") / f"{file.path}{extension}"

        # Basic assertions
        assert local_path.exists(), f"Expected synced file at {local_path}"
        assert local_path.parent.exists(), f"Expected directory at {local_path.parent}"

        # Verify it's not empty
        content = local_path.read_text()
        assert content, f"File at {local_path} should not be empty"
