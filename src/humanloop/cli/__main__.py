import click
import logging
from pathlib import Path
from typing import Optional, Callable
from functools import wraps
from dotenv import load_dotenv, find_dotenv
import os
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
                    "No API key found. Set HUMANLOOP_API_KEY in .env file or environment, or use --api-key"
                )

    return Humanloop(api_key=api_key, base_url=base_url)

def common_options(f: Callable) -> Callable:
    """Decorator for common CLI options."""
    @click.option(
        "--api-key",
        help="Humanloop API key. If not provided, uses HUMANLOOP_API_KEY from .env or environment.",
        default=None,
    )
    @click.option(
        "--env-file",
        help="Path to .env file. If not provided, looks for .env in current directory.",
        default=None,
        type=click.Path(exists=True),
    )
    @click.option(
        "--base-dir",
        help="Base directory for synced files",
        default="humanloop",
        type=click.Path(),
    )
    # Hidden option for internal use - allows overriding the Humanloop API base URL
    # Can be set via --base-url or HUMANLOOP_BASE_URL environment variable
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
            logger.error(f"Error during sync operation: {str(e)}")
            raise click.ClickException(str(e))
    return wrapper

@click.group()
def cli():
    """Humanloop CLI for managing sync operations."""
    pass

@cli.command()
@click.option(
    "--path",
    "-p",
    help="Path to pull (file or directory). If not provided, pulls everything.",
    default=None,
)
@click.option(
    "--environment",
    "-e",
    help="Environment to pull from (e.g. 'production', 'staging')",
    default=None,
)
@common_options
@handle_sync_errors
def pull(path: Optional[str], environment: Optional[str], api_key: Optional[str], env_file: Optional[str], base_dir: str, base_url: Optional[str]):
    """Pull files from Humanloop to local filesystem.
    
    If PATH is provided and ends with .prompt or .agent, pulls that specific file.
    Otherwise, pulls all files under the specified directory path.
    If no PATH is provided, pulls all files from the root.
    """
    client = get_client(api_key, env_file, base_url)
    sync_client = SyncClient(client, base_dir=base_dir)
    
    click.echo("Pulling files from Humanloop...")
    
    click.echo(f"Path: {path or '(root)'}")
    click.echo(f"Environment: {environment or '(default)'}")
        
    successful_files = sync_client.pull(path, environment)
    
    # Get metadata about the operation
    metadata = sync_client.metadata.get_last_operation()
    if metadata:
        click.echo(f"\nSync completed in {metadata['duration_ms']}ms")
        if metadata['successful_files']:
            click.echo(f"\nSuccessfully synced {len(metadata['successful_files'])} files:")
            for file in metadata['successful_files']:
                click.echo(f"  ✓ {file}")
        if metadata['failed_files']:
            click.echo(f"\nFailed to sync {len(metadata['failed_files'])} files:")
            for file in metadata['failed_files']:
                click.echo(f"  ✗ {file}")

@cli.command()
@common_options
@handle_sync_errors
def history(api_key: Optional[str], env_file: Optional[str], base_dir: str, base_url: Optional[str]):
    """Show sync operation history."""
    client = get_client(api_key, env_file, base_url)
    sync_client = SyncClient(client, base_dir=base_dir)
    
    history = sync_client.metadata.get_history()
    if not history:
        click.echo("No sync operations found in history.")
        return
        
    click.echo("Sync Operation History:")
    click.echo("======================")
    
    for op in history:
        click.echo(f"\nOperation: {op['operation_type']}")
        click.echo(f"Timestamp: {op['timestamp']}")
        click.echo(f"Path: {op['path'] or '(root)'}")
        if op['environment']:
            click.echo(f"Environment: {op['environment']}")
        click.echo(f"Duration: {op['duration_ms']}ms")
        if op['successful_files']:
            click.echo(f"Successfully synced {len(op['successful_files'])} file{'' if len(op['successful_files']) == 1 else 's'}")
        if op['failed_files']:
            click.echo(f"Failed to sync {len(op['failed_files'])} file{'' if len(op['failed_files']) == 1 else 's'}")
        if op['error']:
            click.echo(f"Error: {op['error']}")
        click.echo("----------------------")

if __name__ == "__main__":
    cli() 