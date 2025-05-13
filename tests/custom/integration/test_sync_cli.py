from pathlib import Path
from unittest import mock
import pytest
from click.testing import CliRunner
from humanloop.cli.__main__ import cli
from tests.custom.types import SyncableFile


@pytest.fixture
def no_env_file_loading():
    """Fixture that prevents loading API keys from any .env files.

    Use this fixture in tests that verify behavior when no .env files should
    be processed, regardless of whether they exist or not.
    """
    # Prevent any .env file from being loaded
    with mock.patch("humanloop.cli.__main__.load_dotenv", lambda *args, **kwargs: None):
        yield


def test_pull_without_api_key(cli_runner: CliRunner, no_humanloop_api_key_in_env, no_env_file_loading):
    """GIVEN no API key in environment
    WHEN running pull command
    THEN it should fail with appropriate error message
    """
    # WHEN running pull command
    result = cli_runner.invoke(cli, ["pull", "--local-files-directory", "humanloop"])

    # THEN it should fail with appropriate error message
    assert result.exit_code == 1  # Our custom error code for API key issues
    assert "No API key found" in result.output
    assert "Set HUMANLOOP_API_KEY in .env file or environment" in result.output


def test_pull_basic(
    cli_runner: CliRunner,
    syncable_files_fixture: list[SyncableFile],
    tmp_path: Path,  # this path is used as a temporary store for files locally
):
    # GIVEN a base directory for pulled files
    base_dir = str(tmp_path / "humanloop")

    # WHEN running pull command
    result = cli_runner.invoke(cli, ["pull", "--local-files-directory", base_dir, "--verbose"])

    # THEN it should succeed
    assert result.exit_code == 0
    assert "Pulling files from Humanloop..." in result.output
    assert "Pull completed" in result.output

    # THEN the files should exist locally
    for file in syncable_files_fixture:
        extension = f".{file.type}"
        local_path = Path(base_dir) / f"{file.path}{extension}"
        assert local_path.exists(), f"Expected synced file at {local_path}"
        assert local_path.parent.exists(), f"Expected directory at {local_path.parent}"
        assert local_path.read_text(), f"File at {local_path} should not be empty"


def test_pull_with_specific_path(
    cli_runner: CliRunner,
    syncable_files_fixture: list[SyncableFile],
    tmp_path: Path,
):
    """GIVEN a specific path to pull
    WHEN running pull command with path
    THEN it should pull only files from that path
    """
    # GIVEN a base directory and specific path
    base_dir = str(tmp_path / "humanloop")
    test_path = syncable_files_fixture[
        0
    ].path.split(
        "/"
    )[
        0
    ]  # Retrieve the prefix of the first file's path which corresponds to the sdk_test_dir used within syncable_files_fixture

    # WHEN running pull command with path
    result = cli_runner.invoke(cli, ["pull", "--local-files-directory", base_dir, "--path", test_path, "--verbose"])

    # THEN it should succeed and show the path
    assert result.exit_code == 0
    assert f"Path: {test_path}" in result.output

    # THEN only files from that path should exist locally
    for file in syncable_files_fixture:
        extension = f".{file.type}"
        local_path = Path(base_dir) / f"{file.path}{extension}"
        if file.path.startswith(test_path):
            assert local_path.exists(), f"Expected synced file at {local_path}"
        else:
            assert not local_path.exists(), f"Unexpected file at {local_path}"


def test_pull_with_environment(
    cli_runner: CliRunner,
    syncable_files_fixture: list[SyncableFile],
    tmp_path: Path,
):
    # GIVEN a base directory and environment
    base_dir = str(tmp_path / "humanloop")
    environment = "staging"

    # WHEN running pull command with environment
    result = cli_runner.invoke(
        cli,
        [
            "pull",
            "--local-files-directory",
            base_dir,
            "--environment",
            environment,
            "--verbose",
        ],
    )

    # THEN it should succeed and show the environment
    assert result.exit_code == 0
    assert f"Environment: {environment}" in result.output


def test_pull_with_quiet_mode(
    cli_runner: CliRunner,
    syncable_files_fixture: list[SyncableFile],
    tmp_path: Path,
):
    # GIVEN a base directory and quiet mode
    base_dir = str(tmp_path / "humanloop")

    # WHEN running pull command with quiet mode
    result = cli_runner.invoke(cli, ["pull", "--local-files-directory", base_dir, "--quiet"])

    # THEN it should succeed but not show file list
    assert result.exit_code == 0
    assert "Successfully pulled" not in result.output

    # THEN files should still be pulled
    for file in syncable_files_fixture:
        extension = f".{file.type}"
        local_path = Path(base_dir) / f"{file.path}{extension}"
        assert local_path.exists(), f"Expected synced file at {local_path}"


def test_pull_with_invalid_path(
    cli_runner: CliRunner,
):
    # GIVEN an invalid base directory
    path = "nonexistent/path"

    # WHEN running pull command
    result = cli_runner.invoke(cli, ["pull", "--path", path])

    # THEN it should fail
    assert result.exit_code == 1
    assert "Error" in result.output


def test_pull_with_invalid_environment(cli_runner: CliRunner, tmp_path: Path):
    # GIVEN an invalid environment
    environment = "nonexistent"
    base_dir = str(tmp_path / "humanloop")

    # WHEN running pull command
    result = cli_runner.invoke(
        cli,
        [
            "pull",
            "--local-files-directory",
            base_dir,
            "--environment",
            environment,
            "--verbose",
        ],
    )

    # THEN it should fail
    assert result.exit_code == 1
    assert "Error" in result.output
