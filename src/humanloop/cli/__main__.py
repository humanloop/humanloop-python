import click
import logging
from pathlib import Path
from typing import Optional, Callable
from functools import wraps
from dotenv import load_dotenv, find_dotenv
import os
import sys
from humanloop import Humanloop
from humanloop.sync.sync_client import SyncClient
from datetime import datetime

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

MAX_FILES_TO_DISPLAY = 10

def get_client(api_key: Optional[str] = None, env_file: Optional[str] = None, base_url: Optional[str] = None) -> Humanloop:
    """Get a Humanloop client instance.
    
    If no API key is provided, it will be loaded from the .env file, or the environment variable HUMANLOOP_API_KEY.

    Raises:
        click.ClickException: If no API key is found.
    """
    if not api_key:
        if env_file:
            load_dotenv(env_file)
        else:
            env_path = find_dotenv()
            if env_path:
                load_dotenv(env_path)
            else:
                if os.path.exists(".env"):
                    load_dotenv(".env")
                else:
                    load_dotenv()
            
            api_key = os.getenv("HUMANLOOP_API_KEY")
            if not api_key:
                raise click.ClickException(
                    click.style("No API key found. Set HUMANLOOP_API_KEY in .env file or environment, or use --api-key", fg=ERROR_COLOR)
                )

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
        "--base-dir",
        help="Base directory for pulled files",
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
    """Decorator for handling sync operation errors."""
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
    }
)
def cli():
    """Humanloop CLI for managing sync operations."""
    pass

@cli.command()
@click.option(
    "--path",
    "-p",
    help="Path to pull (file or directory). If not provided, pulls everything. "
    "To pull a specific file, ensure the extension for the file is included (e.g. .prompt or .agent). "
    "To pull a directory, simply specify the path to the directory (e.g. abc/def to pull all files under abc/def and its subdirectories).",
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
@handle_sync_errors
@common_options
def pull(
    path: Optional[str], 
    environment: Optional[str], 
    api_key: Optional[str], 
    env_file: Optional[str], 
    base_dir: str, 
    base_url: Optional[str], 
    verbose: bool
):
    """Pull prompt and agent files from Humanloop to your local filesystem.

    \b
    This command will:
    1. Fetch prompt and agent files from your Humanloop workspace
    2. Save them to your local filesystem
    3. Maintain the same directory structure as in Humanloop
    4. Add appropriate file extensions (.prompt or .agent)

    \b
    The files will be saved with the following structure:
    {base_dir}/
    ├── prompts/
    │   ├── my_prompt.prompt
    │   └── nested/
    │       └── another_prompt.prompt
    └── agents/
        └── my_agent.agent

    The operation will overwrite existing files with the latest version from Humanloop
    but will not delete local files that don't exist in the remote workspace.

    Currently only supports syncing prompt and agent files. Other file types will be skipped."""
    client = get_client(api_key, env_file, base_url)
    sync_client = SyncClient(client, base_dir=base_dir, log_level=logging.DEBUG if verbose else logging.WARNING)
    
    click.echo(click.style("Pulling files from Humanloop...", fg=INFO_COLOR))
    click.echo(click.style(f"Path: {path or '(root)'}", fg=INFO_COLOR))
    click.echo(click.style(f"Environment: {environment or '(default)'}", fg=INFO_COLOR))
    
    successful_files = sync_client.pull(path, environment)
    
    # Get metadata about the operation
    metadata = sync_client.metadata.get_last_operation()
    if metadata:
        # Determine if the operation was successful based on failed_files
        is_successful = not metadata.get('failed_files') and not metadata.get('error')
        duration_color = SUCCESS_COLOR if is_successful else ERROR_COLOR
        click.echo(click.style(f"Pull completed in {metadata['duration_ms']}ms", fg=duration_color))
        
        if metadata['successful_files']:
            click.echo(click.style(f"\nSuccessfully pulled {len(metadata['successful_files'])} files:", fg=SUCCESS_COLOR))
            
            if verbose: 
                for file in metadata['successful_files']:   
                    click.echo(click.style(f"  ✓ {file}", fg=SUCCESS_COLOR))
            else:
                files_to_display = metadata['successful_files'][:MAX_FILES_TO_DISPLAY]
                for file in files_to_display:
                    click.echo(click.style(f"  ✓ {file}", fg=SUCCESS_COLOR))

                if len(metadata['successful_files']) > MAX_FILES_TO_DISPLAY:
                    remaining = len(metadata['successful_files']) - MAX_FILES_TO_DISPLAY
                    click.echo(click.style(f"  ...and {remaining} more", fg=SUCCESS_COLOR))
        if metadata['failed_files']:
            click.echo(click.style(f"\nFailed to pull {len(metadata['failed_files'])} files:", fg=ERROR_COLOR))
            for file in metadata['failed_files']:
                click.echo(click.style(f"  ✗ {file}", fg=ERROR_COLOR))
        if metadata.get('error'):
            click.echo(click.style(f"\nError: {metadata['error']}", fg=ERROR_COLOR))
            

if __name__ == "__main__":
    cli() 