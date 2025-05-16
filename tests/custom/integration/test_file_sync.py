from pathlib import Path

import pytest

from humanloop.error import HumanloopRuntimeError
from tests.custom.types import GetHumanloopClientFn, SyncableFile


def test_pull_basic(
    syncable_files_fixture: list[SyncableFile],
    get_humanloop_client: GetHumanloopClientFn,
    tmp_path: Path,
):
    """Test basic file pulling from remote to local filesystem."""
    # GIVEN a set of files in the remote system (from syncable_files_fixture)
    humanloop_client = get_humanloop_client(local_files_directory=str(tmp_path))

    # WHEN running the pull operation
    humanloop_client.pull()

    # THEN our local filesystem should mirror the remote filesystem in the HL Workspace
    for file in syncable_files_fixture:
        extension = f".{file.type}"
        local_path = tmp_path / f"{file.path}{extension}"

        # THEN the file and its directory should exist
        assert local_path.exists(), f"Expected synced file at {local_path}"
        assert local_path.parent.exists(), f"Expected directory at {local_path.parent}"

        # THEN the file should not be empty
        content = local_path.read_text()
        assert content, f"File at {local_path} should not be empty"


def test_pull_with_invalid_path(
    get_humanloop_client: GetHumanloopClientFn,
    sdk_test_dir: str,
    tmp_path: Path,
):
    """Test error handling when path doesn't exist."""
    humanloop_client = get_humanloop_client(local_files_directory=str(tmp_path))
    non_existent_path = f"{sdk_test_dir}/non_existent_directory"

    # Note: This test currently relies on the specific error message from list_files().
    # If implementing explicit directory validation in the future, this test may need updating.
    with pytest.raises(HumanloopRuntimeError, match=f"Directory `{non_existent_path}` does not exist"):
        humanloop_client.pull(path=non_existent_path)


def test_pull_with_invalid_environment(
    get_humanloop_client: GetHumanloopClientFn,
    tmp_path: Path,
):
    """Test error handling when environment doesn't exist."""
    humanloop_client = get_humanloop_client(local_files_directory=str(tmp_path))
    with pytest.raises(HumanloopRuntimeError, match="Environment .* does not exist"):
        humanloop_client.pull(environment="invalid_environment")


# def test_pull_with_environment(
#     get_humanloop_client: GetHumanloopClientFn,
#     syncable_files_fixture: list[SyncableFile],
# ):
#     """Test pulling files filtered by a specific environment."""
#     # NOTE: This test is currently not feasible to implement because:
#     # 1. We have no way of deploying to an environment using its name, only by ID
#     # 2. There's no API endpoint to retrieve environments for an organization
#     #
#     # If implemented, this test would:
#     # 1. Deploy one of the syncable files to a specific environment (e.g., "production" as it's non-default)
#     # 2. Pull files filtering by the production environment
#     # 3. Check if the deployed file is present in the local filesystem
#     # 4. Verify that none of the other syncable files (that weren't deployed to production) are present
#     #    This would confirm that environment filtering works correctly


def test_pull_with_path_filter(
    get_humanloop_client: GetHumanloopClientFn,
    syncable_files_fixture: list[SyncableFile],
    sdk_test_dir: str,
    tmp_path: Path,
):
    """Test that filtering by path correctly limits which files are pulled."""
    # GIVEN a client
    humanloop_client = get_humanloop_client(local_files_directory=str(tmp_path))

    # WHEN pulling only files from the sdk_test_dir path
    humanloop_client.pull(path=sdk_test_dir)

    # THEN count the total number of files pulled
    pulled_file_count = 0
    for path in tmp_path.glob("**/*"):
        if path.is_file():
            # Check that the file is not empty
            content = path.read_text()
            assert content, f"File at {path} should not be empty"
            pulled_file_count += 1

    # The count should match our fixture length
    assert pulled_file_count == len(syncable_files_fixture), (
        f"Expected {len(syncable_files_fixture)} files, got {pulled_file_count}"
    )
