from typing import Generator, List, NamedTuple, Union
from pathlib import Path
import pytest
from humanloop import Humanloop, FileType, AgentResponse, PromptResponse
from humanloop.prompts.client import PromptsClient
from humanloop.agents.client import AgentsClient
from humanloop.error import HumanloopRuntimeError
from tests.custom.types import GetHumanloopClientFn
import logging


class SyncableFile(NamedTuple):
    path: str
    type: FileType
    model: str
    id: str = ""
    version_id: str = ""


@pytest.fixture
def test_file_structure(
    get_humanloop_client: GetHumanloopClientFn,
    sdk_test_dir: str,
) -> Generator[list[SyncableFile], None, None]:
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

    humanloop_client: Humanloop = get_humanloop_client()

    # Create the files in Humanloop
    created_files = []
    for file in files:
        full_path = f"{sdk_test_dir}/{file.path}"
        response: Union[AgentResponse, PromptResponse]
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
        created_files.append(
            SyncableFile(
                path=full_path, type=file.type, model=file.model, id=response.id, version_id=response.version_id
            )
        )

    yield created_files


@pytest.fixture
def cleanup_local_files():
    """Cleanup any locally synced files after tests"""
    yield
    # Clean up the local humanloop directory after tests
    local_dir = Path("humanloop")
    if local_dir.exists():
        import shutil

        shutil.rmtree(local_dir)


def test_pull_basic(
    get_humanloop_client: GetHumanloopClientFn,
    test_file_structure: List[SyncableFile],
):
    """Test that humanloop.sync() correctly syncs remote files to local filesystem"""
    # Run the sync
    humanloop_client = get_humanloop_client()
    successful_files = humanloop_client.pull()

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


def test_overload_with_local_files(
    get_humanloop_client: GetHumanloopClientFn,
    test_file_structure: List[SyncableFile],
):
    """Test that overload_with_local_files correctly handles local files.

    Flow:
    1. Create files in remote (via test_file_structure fixture)
    2. Pull files locally
    3. Test using the pulled files
    """
    # First pull the files locally
    humanloop_client = get_humanloop_client(use_local_files=True)
    humanloop_client.pull()

    # Test using the pulled files
    test_file = test_file_structure[0]  # Use the first test file
    extension = f".{test_file.type}"
    local_path = Path("humanloop") / f"{test_file.path}{extension}"

    # Verify the file was pulled correctly
    assert local_path.exists(), f"Expected pulled file at {local_path}"
    assert local_path.parent.exists(), f"Expected directory at {local_path.parent}"

    # Test call with pulled file
    response: Union[AgentResponse, PromptResponse]
    if test_file.type == "prompt":
        response = humanloop_client.prompts.call(  # type: ignore [assignment]
            path=test_file.path, messages=[{"role": "user", "content": "Testing"}]
        )
        assert response is not None
    elif test_file.type == "agent":
        response = humanloop_client.agents.call(  # type: ignore [assignment]
            path=test_file.path, messages=[{"role": "user", "content": "Testing"}]
        )
        assert response is not None

    # Test with invalid path
    with pytest.raises(HumanloopRuntimeError):
        sub_client: Union[PromptsClient, AgentsClient]
        match test_file.type:
            case "prompt":
                sub_client = humanloop_client.prompts
            case "agent":
                sub_client = humanloop_client.agents
            case _:
                raise ValueError(f"Invalid file type: {test_file.type}")
        sub_client.call(path="invalid/path")


def test_overload_log_with_local_files(
    get_humanloop_client: GetHumanloopClientFn,
    test_file_structure: List[SyncableFile],
    sdk_test_dir: str,
):
    """Test that overload_with_local_files correctly handles local files for log operations.

    Flow:
    1. Create files in remote (via test_file_structure fixture)
    2. Pull files locally
    3. Test logging using the pulled files

    :param humanloop_test_client: The Humanloop client with local files enabled
    :param test_file_structure: List of test files created in remote
    :param cleanup_local_files: Fixture to clean up local files after test
    """
    # First pull the files locally
    humanloop_client = get_humanloop_client(use_local_files=True)
    humanloop_client.pull()

    # Test using the pulled files
    test_file = test_file_structure[0]  # Use the first test file
    extension = f".{test_file.type}"
    local_path = Path("humanloop") / f"{test_file.path}{extension}"

    # Verify the file was pulled correctly
    assert local_path.exists(), f"Expected pulled file at {local_path}"
    assert local_path.parent.exists(), f"Expected directory at {local_path.parent}"

    # Test log with pulled file
    if test_file.type == "prompt":
        response = humanloop_client.prompts.log(  # type: ignore [assignment]
            path=test_file.path, messages=[{"role": "user", "content": "Testing"}], output="Test response"
        )
        assert response is not None
    elif test_file.type == "agent":
        response = humanloop_client.agents.log(  # type: ignore [assignment]
            path=test_file.path, messages=[{"role": "user", "content": "Testing"}], output="Test response"
        )
        assert response is not None

    # Test with invalid path
    with pytest.raises(HumanloopRuntimeError):
        if test_file.type == "prompt":
            humanloop_client.prompts.log(
                path=f"{sdk_test_dir}/invalid/path",
                messages=[{"role": "user", "content": "Testing"}],
                output="Test response",
            )
        elif test_file.type == "agent":
            humanloop_client.agents.log(
                path=f"{sdk_test_dir}/invalid/path",
                messages=[{"role": "user", "content": "Testing"}],
                output="Test response",
            )


def test_overload_version_environment_handling(
    caplog: pytest.LogCaptureFixture,
    get_humanloop_client: GetHumanloopClientFn,
    test_file_structure: List[SyncableFile],
):
    """Test that overload_with_local_files correctly handles version_id and environment parameters.

    Flow:
    1. Create files in remote (via test_file_structure fixture)
    2. Pull files locally
    3. Test that version_id/environment parameters cause remote usage with warning
    """
    # First pull the files locally
    humanloop_client = get_humanloop_client(use_local_files=True)
    humanloop_client.pull()

    # Test using the pulled files
    test_file = test_file_structure[0]  # Use the first test file
    extension = f".{test_file.type}"
    local_path = Path("humanloop") / f"{test_file.path}{extension}"

    # Verify the file was pulled correctly
    assert local_path.exists(), f"Expected pulled file at {local_path}"
    assert local_path.parent.exists(), f"Expected directory at {local_path.parent}"

    # Test with version_id - should use remote with warning
    # Check that the warning was logged
    with caplog.at_level(level=logging.WARNING, logger="humanloop.sdk"):
        if test_file.type == "prompt":
            response = humanloop_client.prompts.call(
                path=test_file.path,
                version_id=test_file.version_id,
                messages=[{"role": "user", "content": "Testing"}],
            )
        elif test_file.type == "agent":
            response = humanloop_client.agents.call(  # type: ignore [assignment]
                path=test_file.path,
                version_id=test_file.version_id,
                messages=[{"role": "user", "content": "Testing"}],
            )
        assert response is not None
        assert any(["Ignoring local file" in record.message for record in caplog.records])

    # Test with environment - should use remote with warning
    if test_file.type == "prompt":
        response = humanloop_client.prompts.call(  # type: ignore [assignment]
            path=test_file.path, environment="production", messages=[{"role": "user", "content": "Testing"}]
        )
    elif test_file.type == "agent":
        response = humanloop_client.agents.call(  # type: ignore [assignment]
            path=test_file.path, environment="production", messages=[{"role": "user", "content": "Testing"}]
        )
    assert response is not None

    if test_file.type == "prompt":
        response = humanloop_client.prompts.call(  # type: ignore [assignment]
            path=test_file.path,
            version_id=test_file.version_id,
            environment="staging",
            messages=[{"role": "user", "content": "Testing"}],
        )
    elif test_file.type == "agent":
        response = humanloop_client.agents.call(  # type: ignore [assignment]
            path=test_file.path,
            version_id=test_file.version_id,
            environment="staging",
            messages=[{"role": "user", "content": "Testing"}],
        )
    assert response is not None
