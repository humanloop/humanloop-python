import logging
import os
import sys
import time
from functools import wraps
from typing import Callable, Optional

import click
from dotenv import load_dotenv

from humanloop import Humanloop
from humanloop.sync.sync_client import SyncClient

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set back to INFO level
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")  # Simplified formatter
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(console_handler)

# Color constants
SUCCESS_COLOR = "green"
ERROR_COLOR = "red"
INFO_COLOR = "blue"
WARNING_COLOR = "yellow"


def load_api_key(env_file: Optional[str] = None) -> str:
    """Load API key from .env file or environment variable.

    Args:
        env_file: Optional path to .env file

    Returns:
        str: The loaded API key

    Raises:
        click.ClickException: If no API key is found
    """
    # Try specific .env file if provided, otherwise default to .env in current directory
    if env_file:
        if not load_dotenv(env_file):  # load_dotenv returns False if file not found/invalid
            raise click.ClickException(
                click.style(
                    f"Failed to load environment file: {env_file} (file not found or invalid format)",
                    fg=ERROR_COLOR,
                )
            )
    else:
        load_dotenv()  # Attempt to load from default .env in current directory

    # Get API key from environment
    api_key = os.getenv("HUMANLOOP_API_KEY")
    if not api_key:
        raise click.ClickException(
            click.style(
                "No API key found. Set HUMANLOOP_API_KEY in .env file or environment, or use --api-key", fg=ERROR_COLOR
            )
        )

    return api_key


def get_client(
    api_key: Optional[str] = None, env_file: Optional[str] = None, base_url: Optional[str] = None
) -> Humanloop:
    """Instantiate a Humanloop client for the CLI.

    Args:
        api_key: Optional API key provided directly
        env_file: Optional path to .env file
        base_url: Optional base URL for the API

    Returns:
        Humanloop: Configured client instance

    Raises:
        click.ClickException: If no API key is found
    """
    if not api_key:
        api_key = load_api_key(env_file)
    print(api_key)
    return Humanloop(api_key=api_key, base_url=base_url)


def common_options(f: Callable) -> Callable:
    """Decorator for common CLI options."""

    @click.option(
        "--api-key",
        help="Humanloop API key. If not provided, uses HUMANLOOP_API_KEY from .env or environment.",
        default=None,
        show_default=False,
    )
    @click.option(
        "--env-file",
        help="Path to .env file. If not provided, looks for .env in current directory.",
        default=None,
        type=click.Path(exists=True),
        show_default=False,
    )
    @click.option(
        "--local-files-directory",
        "--local-dir",
        help="Directory (relative to the current working directory) where Humanloop files are stored locally (default: humanloop/).",
        default="humanloop",
        type=click.Path(),
    )
    @click.option(
        "--base-url",
        default=None,
        hidden=True,
    )
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


def handle_sync_errors(f: Callable) -> Callable:
    """Decorator for handling sync operation errors.

    If an error occurs in any operation that uses this decorator, it will be logged and the program will exit with a non-zero exit code.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            click.echo(click.style(str(f"Error: {e}"), fg=ERROR_COLOR))
            sys.exit(1)

    return wrapper


@click.group(
    help="Humanloop CLI for managing sync operations.",
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 100,
    },
)
def cli():  # Does nothing because used as a group for other subcommands (pull, push, etc.)
    """Humanloop CLI for managing sync operations."""
    pass


@cli.command()
@click.option(
    "--path",
    "-p",
    help="Path in the Humanloop workspace to pull from (file or directory). You can pull an entire directory (e.g. 'my/directory') "
    "or a specific file (e.g. 'my/directory/my_prompt.prompt'). When pulling a directory, all files within that directory and its subdirectories will be included. "
    "Paths should not contain leading or trailing slashes. "
    "If not specified, pulls from the root of the remote workspace.",
    default=None,
)
@click.option(
    "--environment",
    "-e",
    help="Environment to pull from (e.g. 'production', 'staging')",
    default=None,
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed information about the operation",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress output of successful files",
)
@handle_sync_errors
@common_options
def pull(
    path: Optional[str],
    environment: Optional[str],
    api_key: Optional[str],
    env_file: Optional[str],
    local_files_directory: str,
    base_url: Optional[str],
    verbose: bool,
    quiet: bool,
):
    """Pull Prompt and Agent files from Humanloop to your local filesystem.

    \b
    This command will:
    1. Fetch Prompt and Agent files from your Humanloop workspace
    2. Save them to your local filesystem (directory specified by --local-files-directory, default: humanloop/)
    3. Maintain the same directory structure as in Humanloop
    4. Add appropriate file extensions (.prompt or .agent)

    \b
    For example, with the default --local-files-directory=humanloop, files will be saved as:
    ./humanloop/
    ├── my_project/
    │   ├── prompts/
    │   │   ├── my_prompt.prompt
    │   │   └── nested/
    │   │       └── another_prompt.prompt
    │   └── agents/
    │       └── my_agent.agent
    └── another_project/
        └── prompts/
            └── other_prompt.prompt

    \b
    If you specify --local-files-directory=data/humanloop, files will be saved in ./data/humanloop/ instead.

    If a file exists both locally and in the Humanloop workspace, the local file will be overwritten
    with the version from Humanloop. Files that only exist locally will not be affected.

    Currently only supports syncing Prompt and Agent files. Other file types will be skipped."""
    client = get_client(api_key, env_file, base_url)
    # Although pull() is available on the Humanloop client, we instantiate SyncClient separately as we need to control its log level
    sync_client = SyncClient(
        client, base_dir=local_files_directory, log_level=logging.DEBUG if verbose else logging.WARNING
    )

    click.echo(click.style("Pulling files from Humanloop...", fg=INFO_COLOR))
    click.echo(click.style(f"Path: {path or '(root)'}", fg=INFO_COLOR))
    click.echo(click.style(f"Environment: {environment or '(default)'}", fg=INFO_COLOR))

    start_time = time.time()
    successful_files, failed_files = sync_client.pull(path, environment)
    duration_ms = int((time.time() - start_time) * 1000)

    # Determine if the operation was successful based on failed_files
    is_successful = not failed_files
    duration_color = SUCCESS_COLOR if is_successful else ERROR_COLOR
    click.echo(click.style(f"Pull completed in {duration_ms}ms", fg=duration_color))

    if successful_files and not quiet:
        click.echo(click.style(f"\nSuccessfully pulled {len(successful_files)} files:", fg=SUCCESS_COLOR))
        for file in successful_files:
            click.echo(click.style(f"  ✓ {file}", fg=SUCCESS_COLOR))

    if failed_files:
        click.echo(click.style(f"\nFailed to pull {len(failed_files)} files:", fg=ERROR_COLOR))
        for file in failed_files:
            click.echo(click.style(f"  ✗ {file}", fg=ERROR_COLOR))


if __name__ == "__main__":
    cli()
