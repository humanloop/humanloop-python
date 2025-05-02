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

def get_client(api_key: Optional[str] = None, env_file: Optional[str] = None, base_url: Optional[str] = None) -> Humanloop:
    """Get a Humanloop client instance."""
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
        help="Base directory for synced files",
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
@handle_sync_errors
@common_options
def pull(path: Optional[str], environment: Optional[str], api_key: Optional[str], env_file: Optional[str], base_dir: str, base_url: Optional[str]):
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
    sync_client = SyncClient(client, base_dir=base_dir)
    
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
        click.echo(click.style(f"\nSync completed in {metadata['duration_ms']}ms", fg=duration_color))
        
        if metadata['successful_files']:
            click.echo(click.style(f"\nSuccessfully synced {len(metadata['successful_files'])} files:", fg=SUCCESS_COLOR))
            for file in metadata['successful_files']:
                click.echo(click.style(f"  ✓ {file}", fg=SUCCESS_COLOR))
        if metadata['failed_files']:
            click.echo(click.style(f"\nFailed to sync {len(metadata['failed_files'])} files:", fg=ERROR_COLOR))
            for file in metadata['failed_files']:
                click.echo(click.style(f"  ✗ {file}", fg=ERROR_COLOR))
        if metadata.get('error'):
            click.echo(click.style(f"\nError: {metadata['error']}", fg=ERROR_COLOR))

def format_timestamp(timestamp: str) -> str:
    """Format timestamp to a more readable format."""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, AttributeError):
        return timestamp

@cli.command()
@click.option(
    "--oneline",
    is_flag=True,
    help="Display history in a single line per operation",
)
@handle_sync_errors
@common_options
def history(api_key: Optional[str], env_file: Optional[str], base_dir: str, base_url: Optional[str], oneline: bool):
    """Show sync operation history."""
    client = get_client(api_key, env_file, base_url)
    sync_client = SyncClient(client, base_dir=base_dir)
    
    history = sync_client.metadata.get_history()
    if not history:
        click.echo(click.style("No sync operations found in history.", fg=WARNING_COLOR))
        return
        
    if not oneline:
        click.echo(click.style("Sync Operation History:", fg=INFO_COLOR))
        click.echo(click.style("======================", fg=INFO_COLOR))
    
    for op in history:
        if oneline:
            # Format: timestamp | operation_type | path | environment | duration_ms | status
            status = click.style("✓", fg=SUCCESS_COLOR) if not op['failed_files'] else click.style("✗", fg=ERROR_COLOR)
            click.echo(f"{format_timestamp(op['timestamp'])} | {op['operation_type']} | {op['path'] or '(root)'} | {op['environment'] or '-'} | {op['duration_ms']}ms | {status}")
        else:
            click.echo(click.style(f"\nOperation: {op['operation_type']}", fg=INFO_COLOR))
            click.echo(f"Timestamp: {format_timestamp(op['timestamp'])}")
            click.echo(f"Path: {op['path'] or '(root)'}")
            if op['environment']:
                click.echo(f"Environment: {op['environment']}")
            click.echo(f"Duration: {op['duration_ms']}ms")
            if op['successful_files']:
                click.echo(click.style(f"Successfully synced {len(op['successful_files'])} file{'' if len(op['successful_files']) == 1 else 's'}", fg=SUCCESS_COLOR))
            if op['failed_files']:
                click.echo(click.style(f"Failed to sync {len(op['failed_files'])} file{'' if len(op['failed_files']) == 1 else 's'}", fg=ERROR_COLOR))
            if op['error']:
                click.echo(click.style(f"Error: {op['error']}", fg=ERROR_COLOR))
            click.echo(click.style("----------------------", fg=INFO_COLOR))

if __name__ == "__main__":
    cli() 