from pathlib import Path

import pytest

from humanloop.error import HumanloopRuntimeError
from tests.custom.types import GetHumanloopClientFn, SyncableFile


def test_pull_basic(
    syncable_files_fixture: list[SyncableFile],
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


def test_pull_with_invalid_path(
    get_humanloop_client: GetHumanloopClientFn,
    sdk_test_dir: str,
):
    """Test that humanloop.sync() raises an error when the path is invalid"""
    humanloop_client = get_humanloop_client()
    non_existent_path = f"{sdk_test_dir}/non_existent_directory"

    # Note: This test currently relies on the specific error message from list_files().
    # If implementing explicit directory validation in the future, this test may need updating.
    with pytest.raises(HumanloopRuntimeError, match=f"Directory `{non_existent_path}` does not exist"):
        humanloop_client.pull(path=non_existent_path)

def test_pull_with_invalid_environment(
    get_humanloop_client: GetHumanloopClientFn,
):
    """Test that humanloop.sync() raises an error when the environment is invalid"""
    humanloop_client = get_humanloop_client()
    with pytest.raises(HumanloopRuntimeError, match="Invalid environment"):
        humanloop_client.pull(environment="invalid_environment")