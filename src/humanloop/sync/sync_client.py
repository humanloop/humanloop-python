import multiprocessing
import os
import logging
from pathlib import Path
import concurrent.futures
from typing import List, TYPE_CHECKING, Union, cast, Optional

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
    """Client for managing synchronization between local filesystem and Humanloop."""
    
    def __init__(
        self, 
        client: "BaseHumanloop",
        base_dir: str = "humanloop",
        max_workers: Optional[int] = None
    ):
        """
        Parameters
        ----------
        client: Humanloop client instance
        base_dir: Base directory for synced files (default: "humanloop")
        max_workers: Maximum number of worker threads (default: CPU count * 2)
        """
        self.client = client
        self.base_dir = Path(base_dir)
        self.max_workers = max_workers or multiprocessing.cpu_count() * 2

    def _save_serialized_file(self, serialized_content: str, file_path: str, file_type: FileType) -> None:
        """Save serialized file to local filesystem.

        Args:
            serialized_content: The content to save
            file_path: The path where to save the file
            file_type: The type of file (prompt or agent)
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
            logger.info(f"Syncing {file_type} {file_path}")
        except Exception as e:
            logger.error(f"Failed to sync {file_type} {file_path}: {str(e)}")
            raise

    def pull(self) -> List[str]:
        """Sync prompt and agent files from Humanloop to local filesystem.

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
                    include_content=True
                )

                if len(response.records) == 0:
                    break

                # Process each file
                for file in response.records:
                    # Skip if not a prompt or agent
                    if file.type not in ["prompt", "agent"]:
                        logger.warning(f"Skipping unsupported file type: {file.type}")
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