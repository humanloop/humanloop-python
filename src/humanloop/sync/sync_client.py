import multiprocessing
import os
import logging
from pathlib import Path
import concurrent.futures
from typing import List, TYPE_CHECKING, Union, cast, Optional, Dict
from functools import lru_cache

from humanloop.types import FileType, PromptResponse, AgentResponse, ToolResponse, DatasetResponse, EvaluatorResponse, FlowResponse
from humanloop.core.api_error import ApiError

if TYPE_CHECKING:
    from humanloop.base_client import BaseHumanloop

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(console_handler)

class SyncClient:
    """Client for managing synchronization between local filesystem and Humanloop.
    
    This client provides file synchronization between Humanloop and the local filesystem,
    with built-in caching for improved performance. The cache uses Python's LRU (Least 
    Recently Used) cache to automatically manage memory usage by removing least recently 
    accessed files when the cache is full.
    
    The cache is automatically updated when files are pulled or saved, and can be
    manually cleared using the clear_cache() method.
    """
    
    def __init__(
        self, 
        client: "BaseHumanloop",
        base_dir: str = "humanloop",
        cache_size: int = 100
    ):
        """
        Parameters
        ----------
        client: Humanloop client instance
        base_dir: Base directory for synced files (default: "humanloop")
        cache_size: Maximum number of files to cache (default: 100)
        """
        self.client = client
        self.base_dir = Path(base_dir)
        self._cache_size = cache_size
        # Create a new cached version of get_file_content with the specified cache size
        self.get_file_content = lru_cache(maxsize=cache_size)(self._get_file_content_impl)

    def _get_file_content_impl(self, path: str, file_type: FileType) -> Optional[str]:
        """Implementation of get_file_content without the cache decorator.
        
        This is the actual implementation that gets wrapped by lru_cache.
        """
        try:
            # Construct path to local file
            local_path = self.base_dir / path
            # Add appropriate extension
            local_path = local_path.parent / f"{local_path.stem}.{file_type}"
            
            if local_path.exists():
                # Read the file content
                with open(local_path) as f:
                    file_content = f.read()
                logger.debug(f"Using local file content from {local_path}")
                return file_content
            else:
                logger.warning(f"Local file not found: {local_path}, falling back to API")
                return None
        except Exception as e:
            logger.error(f"Error reading local file: {e}, falling back to API")
            return None

    def get_file_content(self, path: str, file_type: FileType) -> Optional[str]:
        """Get the content of a file from cache or filesystem.
        
        This method uses an LRU cache to store file contents. When the cache is full,
        the least recently accessed files are automatically removed to make space.
        
        Args:
            path: The normalized path to the file (without extension)
            file_type: The type of file (prompt or agent)
            
        Returns:
            The file content if found, None otherwise
        """
        return self._get_file_content_impl(path, file_type)

    def clear_cache(self) -> None:
        """Clear the LRU cache."""
        self.get_file_content.cache_clear()

    def _normalize_path(self, path: str) -> str:
        """Normalize the path by:
        1. Removing any file extensions (.prompt, .agent)
        2. Converting backslashes to forward slashes
        3. Removing leading and trailing slashes
        4. Removing leading and trailing whitespace
        5. Normalizing multiple consecutive slashes into a single forward slash

        Args:
            path: The path to normalize

        Returns:
            The normalized path
        """
        # Remove any file extensions
        path = path.rsplit('.', 1)[0] if '.' in path else path
        
        # Convert backslashes to forward slashes and normalize multiple slashes
        path = path.replace('\\', '/')
        
        # Remove leading/trailing whitespace and slashes
        path = path.strip().strip('/')
        
        # Normalize multiple consecutive slashes into a single forward slash
        while '//' in path:
            path = path.replace('//', '/')
            
        return path

    def is_file(self, path: str) -> bool:
        """Check if the path is a file by checking for .prompt or .agent extension.

        Args:
            path: The path to check

        Returns:
            True if the path ends with .prompt or .agent, False otherwise
        """
        return path.endswith('.prompt') or path.endswith('.agent')

    def _save_serialized_file(self, serialized_content: str, file_path: str, file_type: FileType) -> None:
        """Save serialized file to local filesystem.

        Args:
            serialized_content: The content to save
            file_path: The path where to save the file
            file_type: The type of file (prompt or agent)

        Raises:
            Exception: If there is an error saving the file
        """
        try:
            # Create full path including base_dir prefix
            full_path = self.base_dir / file_path
            # Create directory if it doesn't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Add file type extension
            new_path = full_path.parent / f"{full_path.stem}.{file_type}"

            # Write content to file
            with open(new_path, "w") as f:
                f.write(serialized_content)
            
            # Clear the cache for this file to ensure we get fresh content next time
            self.clear_cache()
            
            logger.info(f"Syncing {file_type} {file_path}")
        except Exception as e:
            logger.error(f"Failed to sync {file_type} {file_path}: {str(e)}")
            raise

    def _pull_file(self, path: str, environment: str | None = None) -> None:
        """Pull a specific file from Humanloop to local filesystem.

        Args:
            path: The path of the file without the extension (e.g. "path/to/file")
            environment: The environment to pull the file from

        Raises:
            ValueError: If the file type is not supported
            Exception: If there is an error pulling the file
        """
        file = self.client.files.retrieve_by_path(
            path, 
            environment=environment,
            include_content=True
        )

        if file.type not in ["prompt", "agent"]:
            raise ValueError(f"Unsupported file type: {file.type}")

        self._save_serialized_file(file.content, file.path, file.type)

    def _pull_directory(self, 
            path: str | None = None,    
            environment: str | None = None, 
        ) -> List[str]:
        """Sync prompt and agent files from Humanloop to local filesystem.

        If `path` is provided, only the files under that path will be pulled.
        If `environment` is provided, the files will be pulled from that environment.

        Args: 
            path: The path of the directory to pull from (e.g. "path/to/directory")
            environment: The environment to pull the files from

        Returns:
            List of successfully processed file paths
        """
        successful_files = []
        failed_files = []
        page = 1

        while True:
            try:
                response = self.client.files.list_files(
                    type=["prompt", "agent"], 
                    page=page,
                    include_content=True,
                    environment=environment,
                    directory=path
                )

                if len(response.records) == 0:
                    break

                # Process each file
                for file in response.records:
                    # Skip if not a prompt or agent
                    if file.type not in ["prompt", "agent"]:
                        logger.warning(f"Skipping unsupported file type: {file.type}")
                        continue

                    if not file.path.startswith(path):
                        # Filter by path
                        continue

                    # Skip if no content
                    if not getattr(file, "content", None):
                        logger.warning(f"No content found for {file.type} {getattr(file, 'id', '<unknown>')}")
                        continue

                    try:
                        self._save_serialized_file(file.content, file.path, file.type)
                        successful_files.append(file.path)
                    except Exception as e:
                        failed_files.append(file.path)
                        logger.error(f"Task failed for {file.path}: {str(e)}")

                page += 1
            except Exception as e:
                logger.error(f"Failed to fetch page {page}: {str(e)}")
                break

        # Log summary
        if successful_files:
            logger.info(f"\nSynced {len(successful_files)} files")
        if failed_files:
            logger.error(f"Failed to sync {len(failed_files)} files")

        return successful_files

    def pull(self, path: str, environment: str | None = None) -> List[str]:
        """Pull files from Humanloop to local filesystem.

        If the path ends with .prompt or .agent, pulls that specific file.
        Otherwise, pulls all files under the specified directory path.

        Args:
            path: The path to pull from (either a specific file or directory)
            environment: The environment to pull from

        Returns:
            List of successfully processed file paths
        """
        normalized_path = self._normalize_path(path)
        if self.is_file(path):
            self._pull_file(normalized_path, environment)
            return [path]
        else:
            return self._pull_directory(normalized_path, environment)
