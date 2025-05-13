import logging
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING
from functools import lru_cache
import typing
from humanloop.types import FileType
import time
from humanloop.error import HumanloopRuntimeError
import json

if TYPE_CHECKING:
    from humanloop.base_client import BaseHumanloop

# Set up logging
logger = logging.getLogger("humanloop.sdk.sync")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(console_handler)

# Default cache size for file content caching
DEFAULT_CACHE_SIZE = 100


def format_api_error(error: Exception) -> str:
    """Format API error messages to be more user-friendly."""
    error_msg = str(error)
    if "status_code" not in error_msg or "body" not in error_msg:
        return error_msg

    try:
        # Extract the body part and parse as JSON
        body_str = error_msg.split("body: ")[1]
        # Convert Python dict string to valid JSON by:
        # 1. Escaping double quotes
        # 2. Replacing single quotes with double quotes
        body_str = body_str.replace('"', '\\"').replace("'", '"')
        body = json.loads(body_str)

        # Get the detail from the body
        detail = body.get("detail", {})

        # Handle both string and dictionary types for detail
        if isinstance(detail, str):
            return detail
        elif isinstance(detail, dict):
            return detail.get("description") or detail.get("msg") or error_msg
        else:
            return error_msg
    except Exception as e:
        logger.debug(f"Failed to parse error message: {str(e)}")
        return error_msg


SerializableFileType = typing.Literal["prompt", "agent"]


class SyncClient:
    """Client for managing synchronization between local filesystem and Humanloop.

    This client provides file synchronization between Humanloop and the local filesystem,
    with built-in caching for improved performance. The cache uses Python's LRU (Least
    Recently Used) cache to automatically manage memory usage by removing least recently
    accessed files when the cache is full.

    The cache is automatically updated when files are pulled or saved, and can be
    manually cleared using the clear_cache() method.
    """

    # File types that can be serialized to/from the filesystem
    SERIALIZABLE_FILE_TYPES = frozenset(typing.get_args(SerializableFileType))

    def __init__(
        self,
        client: "BaseHumanloop",
        base_dir: str = "humanloop",
        cache_size: int = DEFAULT_CACHE_SIZE,
        log_level: int = logging.WARNING,
    ):
        """
        Parameters
        ----------
        client: Humanloop client instance
        base_dir: Base directory for synced files (default: "humanloop")
        cache_size: Maximum number of files to cache (default: DEFAULT_CACHE_SIZE)
        log_level: Log level for logging (default: WARNING)
        """
        self.client = client
        self.base_dir = Path(base_dir)
        self._cache_size = cache_size

        logger.setLevel(log_level)

        # Create a new cached version of get_file_content with the specified cache size
        self.get_file_content = lru_cache(maxsize=cache_size)(  # type: ignore [assignment]
            self._get_file_content_implementation,
        )

    def _get_file_content_implementation(self, path: str, file_type: SerializableFileType) -> str:
        """Implementation of get_file_content without the cache decorator.

        This is the actual implementation that gets wrapped by lru_cache.

        Args:
            path: The normalized path to the file (without extension)
            file_type: The type of file to get the content of (SerializableFileType)

        Returns:
            The raw file content

        Raises:
            HumanloopRuntimeError: In two cases:
                1. If the file doesn't exist at the expected location
                2. If there's a filesystem error when trying to read the file
                   (e.g., permission denied, file is locked, etc.)
        """
        # Construct path to local file
        local_path = self.base_dir / path
        # Add appropriate extension
        local_path = local_path.parent / f"{local_path.stem}.{file_type}"

        if not local_path.exists():
            raise HumanloopRuntimeError(f"Local file not found: {local_path}")

        try:
            # Read the raw file content
            with open(local_path) as f:
                file_content = f.read()
            logger.debug(f"Using local file content from {local_path}")
            return file_content
        except Exception as e:
            raise HumanloopRuntimeError(f"Error reading local file {local_path}: {str(e)}")

    def get_file_content(self, path: str, file_type: SerializableFileType) -> str:
        """Get the raw file content of a file from cache or filesystem.

        This method uses an LRU cache to store file contents. When the cache is full,
        the least recently accessed files are automatically removed to make space.

        Args:
            path: The normalized path to the file (without extension)
            file_type: The type of file (Prompt or Agent)

        Returns:
            The raw file content

        Raises:
            HumanloopRuntimeError: If the file doesn't exist or can't be read
        """
        return self._get_file_content_implementation(path, file_type)

    def clear_cache(self) -> None:
        """Clear the LRU cache."""
        self.get_file_content.cache_clear()  # type: ignore [attr-defined]

    def _normalize_path(self, path: str) -> str:
        """Normalize the path by:
        1. Converting to a Path object to handle platform-specific separators
        2. Removing any file extensions
        3. Converting to a string with forward slashes and no leading/trailing slashes
        """
        # Convert to Path object to handle platform-specific separators
        path_obj = Path(path)

        # Remove extension, convert to string with forward slashes, and remove leading/trailing slashes
        normalized = str(path_obj.with_suffix(""))
        # Replace all backslashes and normalize multiple forward slashes
        return "/".join(part for part in normalized.replace("\\", "/").split("/") if part)

    def is_file(self, path: str) -> bool:
        """Check if the path is a file by checking for .{file_type} extension for serializable file types."""
        return path.endswith(tuple(f".{file_type}" for file_type in self.SERIALIZABLE_FILE_TYPES))

    def _save_serialized_file(
        self,
        serialized_content: str,
        file_path: str,
        file_type: SerializableFileType,
    ) -> None:
        """Save serialized file to local filesystem."""
        try:
            # Create full path including base_dir prefix
            full_path = self.base_dir / file_path
            # Create directory if it doesn't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Add file type extension
            new_path = full_path.parent / f"{full_path.stem}.{file_type}"

            # Write raw file content to file
            with open(new_path, "w") as f:
                f.write(serialized_content)
        except Exception as e:
            logger.error(f"Failed to write {file_type} {file_path} to disk: {str(e)}")
            raise

    def _pull_file(self, path: str | None = None, environment: str | None = None) -> bool:
        """Pull a specific file from Humanloop to local filesystem.

        Returns:
            True if the file was successfully pulled, False otherwise
        """
        try:
            file = self.client.files.retrieve_by_path(
                path=path,
                environment=environment,
                include_raw_file_content=True,
            )
            
            if file.type not in self.SERIALIZABLE_FILE_TYPES:
                logger.error(f"Unsupported file type: {file.type}")
                return False

            if not file.raw_file_content:  # type: ignore [union-attr]
                logger.error(f"No content found for {file.type} {path}")
                return False

            self._save_serialized_file(
                serialized_content=file.raw_file_content,  # type: ignore [union-attr]
                file_path=file.path,
                file_type=typing.cast(SerializableFileType, file.type),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to pull file {path}: {str(e)}")
            return False

    def _pull_directory(
        self,
        path: str | None = None,
        environment: str | None = None,
    ) -> Tuple[List[str], List[str]]:
        """Sync Prompt and Agent files from Humanloop to local filesystem.

        Returns:
            Tuple of two lists:
            - First list contains paths of successfully synced files
            - Second list contains paths of files that failed to sync. 
              Failures can occur due to missing content in the response or errors during local file writing.

        Raises:
            HumanloopRuntimeError: If there's an error communicating with the API
        """
        successful_files = []
        failed_files = []
        page = 1

        logger.debug(f"Fetching files from directory: {path or '(root)'} in environment: {environment or '(default)'}")

        while True:
            try:
                logger.debug(f"`{path}`: Requesting page {page} of files")
                response = self.client.files.list_files(
                    type=list(self.SERIALIZABLE_FILE_TYPES),
                    page=page,
                    include_raw_file_content=True,
                    environment=environment,
                    path=path,
                )

                if len(response.records) == 0:
                    logger.debug(f"Finished reading files for path `{path}`")
                    break

                logger.debug(f"`{path}`: Read page {page} containing {len(response.records)} files")

                # Process each file
                for file in response.records:
                    # Skip if not a serializable file type
                    if file.type not in self.SERIALIZABLE_FILE_TYPES:
                        logger.warning(f"Skipping unsupported file type: {file.type}")
                        continue

                    file_type: SerializableFileType = typing.cast(
                        SerializableFileType,
                        file.type,
                    )

                    # Skip if no raw file content
                    if not getattr(file, "raw_file_content", None) or not file.raw_file_content:  # type: ignore [union-attr]
                        logger.warning(f"No content found for {file.type} {file.path}")
                        failed_files.append(file.path)
                        continue

                    try:
                        logger.debug(f"Writing {file.type} {file.path} to disk")
                        self._save_serialized_file(
                            serialized_content=file.raw_file_content,  # type: ignore [union-attr]
                            file_path=file.path,
                            file_type=file_type,
                        )
                        successful_files.append(file.path)
                    except Exception as e:
                        failed_files.append(file.path)
                        logger.error(f"Failed to save {file.path}: {str(e)}")

                page += 1
            except Exception as e:
                formatted_error = format_api_error(e)
                raise HumanloopRuntimeError(f"Failed to fetch page {page}: {formatted_error}")

        if successful_files:
            logger.info(f"Successfully pulled {len(successful_files)} files")
        if failed_files:
            logger.warning(f"Failed to pull {len(failed_files)} files")

        return successful_files, failed_files

    def pull(self, path: str | None = None, environment: str | None = None) -> Tuple[List[str], List[str]]:
        """Pull files from Humanloop to local filesystem.

        If the path ends with .prompt or .agent, pulls that specific file.
        Otherwise, pulls all files under the specified path.
        If no path is provided, pulls all files from the root.

        Args:
            path: The path to pull from (either a specific file or directory)
            environment: The environment to pull from

        Returns:
            Tuple of two lists:
            - First list contains paths of successfully synced files
            - Second list contains paths of files that failed to sync

        Raises:
            HumanloopRuntimeError: If there's an error communicating with the API
        """
        start_time = time.time()
        normalized_path = self._normalize_path(path) if path else None

        logger.info(
            f"Starting pull operation: path={normalized_path or '(root)'}, environment={environment or '(default)'}"
        )

        try:
            if path is None:
                # Pull all files from the root
                logger.debug("Pulling all files from root")
                successful_files, failed_files = self._pull_directory(
                    path=None,
                    environment=environment,
                )
            else:
                if self.is_file(path.strip()):
                    logger.debug(f"Pulling file: {normalized_path}")
                    if self._pull_file(path=normalized_path, environment=environment):
                        successful_files = [path]
                        failed_files = []
                    else:
                        successful_files = []
                        failed_files = [path]
                else:
                    logger.debug(f"Pulling directory: {normalized_path}")
                    successful_files, failed_files = self._pull_directory(normalized_path, environment)

            # Clear the cache at the end of each pull operation
            self.clear_cache()

            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Pull completed in {duration_ms}ms: {len(successful_files)} files succeeded")

            return successful_files, failed_files
        except Exception as e:
            raise HumanloopRuntimeError(f"Pull operation failed: {str(e)}")
