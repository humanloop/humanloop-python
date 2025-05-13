from pathlib import Path
from typing import List, Union

import pytest

from humanloop import AgentResponse, PromptResponse
from humanloop.agents.client import AgentsClient
from humanloop.error import HumanloopRuntimeError
from humanloop.prompts.client import PromptsClient
from tests.custom.types import GetHumanloopClientFn, SyncableFile


@pytest.fixture
def cleanup_local_files():
    """Cleanup any locally synced files after tests"""
    yield
    local_dir = Path("humanloop")
    if local_dir.exists():
        import shutil

        shutil.rmtree(local_dir)


def test_pull_basic(
    syncable_files_fixture: List[SyncableFile],
    get_humanloop_client: GetHumanloopClientFn,
):
    """Test that humanloop.sync() correctly syncs remote files to local filesystem"""
    # GIVEN a set of files in the remote system (from syncable_files_fixture)
    humanloop_client = get_humanloop_client()

    # WHEN running the sync
    humanloop_client.pull()

    # THEN our local filesystem should mirror the remote filesystem in the HL Workspace
    for file in syncable_files_fixture:
        extension = f".{file.type}"
        local_path = Path("humanloop") / f"{file.path}{extension}"

        # THEN the file and its directory should exist
        assert local_path.exists(), f"Expected synced file at {local_path}"
        assert local_path.parent.exists(), f"Expected directory at {local_path.parent}"

        # THEN the file should not be empty
        content = local_path.read_text()
        assert content, f"File at {local_path} should not be empty"


def test_overload_with_local_files(
    get_humanloop_client: GetHumanloopClientFn,
    syncable_files_fixture: List[SyncableFile],
):
    """Test that overload_with_local_files correctly handles local files."""
    # GIVEN a client with use_local_files=True and pulled files
    humanloop_client = get_humanloop_client(use_local_files=True)
    humanloop_client.pull()

    # GIVEN a test file from the structure
    test_file = syncable_files_fixture[0]
    extension = f".{test_file.type}"
    local_path = Path("humanloop") / f"{test_file.path}{extension}"

    # THEN the file should exist locally
    assert local_path.exists(), f"Expected pulled file at {local_path}"
    assert local_path.parent.exists(), f"Expected directory at {local_path.parent}"

    # WHEN calling the file
    response: Union[AgentResponse, PromptResponse]
    if test_file.type == "prompt":
        response = humanloop_client.prompts.call(  # type: ignore [assignment]
            path=test_file.path, messages=[{"role": "user", "content": "Testing"}]
        )
    elif test_file.type == "agent":
        response = humanloop_client.agents.call(  # type: ignore [assignment]
            path=test_file.path, messages=[{"role": "user", "content": "Testing"}]
        )
    # THEN the response should not be None
    assert response is not None

    # WHEN calling with an invalid path
    # THEN it should raise HumanloopRuntimeError
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
    syncable_files_fixture: List[SyncableFile],
    sdk_test_dir: str,
):
    """Test that overload_with_local_files correctly handles local files for log operations."""
    # GIVEN a client with use_local_files=True and pulled files
    humanloop_client = get_humanloop_client(use_local_files=True)
    humanloop_client.pull()

    # GIVEN a test file from the structure
    test_file = syncable_files_fixture[0]
    extension = f".{test_file.type}"
    local_path = Path("humanloop") / f"{test_file.path}{extension}"

    # THEN the file should exist locally
    assert local_path.exists(), f"Expected pulled file at {local_path}"
    assert local_path.parent.exists(), f"Expected directory at {local_path.parent}"

    # WHEN logging with the pulled file
    if test_file.type == "prompt":
        response = humanloop_client.prompts.log(  # type: ignore [assignment]
            path=test_file.path, messages=[{"role": "user", "content": "Testing"}], output="Test response"
        )
    elif test_file.type == "agent":
        response = humanloop_client.agents.log(  # type: ignore [assignment]
            path=test_file.path, messages=[{"role": "user", "content": "Testing"}], output="Test response"
        )
    # THEN the response should not be None
    assert response is not None

    # WHEN logging with an invalid path
    # THEN it should raise HumanloopRuntimeError
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
    get_humanloop_client: GetHumanloopClientFn,
    syncable_files_fixture: List[SyncableFile],
):
    """Test that overload_with_local_files correctly handles version_id and environment parameters."""
    # GIVEN a client with use_local_files=True and pulled files
    humanloop_client = get_humanloop_client(use_local_files=True)
    humanloop_client.pull()

    # GIVEN a test file from the structure
    test_file = syncable_files_fixture[0]
    extension = f".{test_file.type}"
    local_path = Path("humanloop") / f"{test_file.path}{extension}"

    # THEN the file should exist locally
    assert local_path.exists(), f"Expected pulled file at {local_path}"
    assert local_path.parent.exists(), f"Expected directory at {local_path.parent}"

    # WHEN calling with version_id
    # THEN it should raise HumanloopRuntimeError
    with pytest.raises(HumanloopRuntimeError, match="Cannot use local file.*version_id or environment was specified"):
        if test_file.type == "prompt":
            humanloop_client.prompts.call(
                path=test_file.path,
                version_id=test_file.version_id,
                messages=[{"role": "user", "content": "Testing"}],
            )
        elif test_file.type == "agent":
            humanloop_client.agents.call(
                path=test_file.path,
                version_id=test_file.version_id,
                messages=[{"role": "user", "content": "Testing"}],
            )

    # WHEN calling with environment
    # THEN it should raise HumanloopRuntimeError
    with pytest.raises(HumanloopRuntimeError, match="Cannot use local file.*version_id or environment was specified"):
        if test_file.type == "prompt":
            humanloop_client.prompts.call(
                path=test_file.path,
                environment="production",
                messages=[{"role": "user", "content": "Testing"}],
            )
        elif test_file.type == "agent":
            humanloop_client.agents.call(
                path=test_file.path,
                environment="production",
                messages=[{"role": "user", "content": "Testing"}],
            )

    # WHEN calling with both version_id and environment
    # THEN it should raise HumanloopRuntimeError
    with pytest.raises(HumanloopRuntimeError, match="Cannot use local file.*version_id or environment was specified"):
        if test_file.type == "prompt":
            humanloop_client.prompts.call(
                path=test_file.path,
                version_id=test_file.version_id,
                environment="staging",
                messages=[{"role": "user", "content": "Testing"}],
            )
        elif test_file.type == "agent":
            humanloop_client.agents.call(
                path=test_file.path,
                version_id=test_file.version_id,
                environment="staging",
                messages=[{"role": "user", "content": "Testing"}],
            )
