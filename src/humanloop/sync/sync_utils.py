import os
import logging
from pathlib import Path
import concurrent.futures
from typing import List, TYPE_CHECKING, Union, cast

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


def _save_serialized_file(serialized_content: str, file_path: str, file_type: FileType) -> None:
    """Save serialized file to local filesystem.

    :param serialized_content: The content to save
    :param file_path: The path where to save the file
    :param file_type: The type of file (prompt or agent)
    """
    try:
        # Create full path including humanloop/ prefix
        full_path = Path("humanloop") / file_path
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


def _process_file(client: "BaseHumanloop", file: Union[PromptResponse, AgentResponse, ToolResponse, DatasetResponse, EvaluatorResponse, FlowResponse]) -> None:
    """Process a single file by serializing and saving it.

    Currently only supports prompt and agent files. Other file types will be skipped.

    :param client: Humanloop client instance
    :param file: The file to process (must be a PromptResponse or AgentResponse)
    """
    try:
        # Skip if not a prompt or agent
        if file.type not in ["prompt", "agent"]:
            logger.warning(f"Skipping unsupported file type: {file.type}")
            return

        # Cast to the correct type for type checking
        if file.type == "prompt":
            file = cast(PromptResponse, file)
        elif file.type == "agent":
            file = cast(AgentResponse, file)

        # Serialize the file based on its type
        try:
            if file.type == "prompt":
                serialized = client.prompts.serialize(id=file.id)
            elif file.type == "agent":
                serialized = client.agents.serialize(id=file.id)
            else:
                logger.warning(f"Skipping unsupported file type: {file.type}")
                return
        except ApiError as e:
            # The SDK returns the YAML content in the error body when it can't parse as JSON
            if e.status_code == 200:
                serialized = e.body
            else:
                raise
        except Exception as e:
            logger.error(f"Failed to serialize {file.type} {file.id}: {str(e)}")
            raise

        # Save to local filesystem
        _save_serialized_file(serialized, file.path, file.type)

    except Exception as e:
        logger.error(f"Error processing file {file.path}: {str(e)}")
        raise


def sync(client: "BaseHumanloop") -> List[str]:
    """Sync prompt and agent files from Humanloop to local filesystem.

    :param client: Humanloop client instance
    :return: List of successfully processed file paths
    """
    successful_files = []
    failed_files = []

    # Create a thread pool for processing files
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        page = 1

        while True:
            try:
                response = client.files.list_files(type=["prompt", "agent"], page=page)

                if len(response.records) == 0:
                    break

                # Submit each file for processing
                for file in response.records:
                    future = executor.submit(_process_file, client, file)
                    futures.append((file.path, future))

                page += 1
            except Exception as e:
                logger.error(f"Failed to fetch page {page}: {str(e)}")
                break

        # Wait for all tasks to complete
        for file_path, future in futures:
            try:
                future.result()
                successful_files.append(file_path)
            except Exception as e:
                failed_files.append(file_path)
                logger.error(f"Task failed for {file_path}: {str(e)}")

    # Log summary
    if successful_files:
        logger.info(f"\nSynced {len(successful_files)} files")
    if failed_files:
        logger.error(f"Failed to sync {len(failed_files)} files")

    return successful_files
