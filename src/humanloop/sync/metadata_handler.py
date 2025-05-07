import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, NotRequired
import logging

logger = logging.getLogger(__name__)

class OperationData(TypedDict):
    """Type definition for operation data structure."""
    timestamp: str
    operation_type: str
    path: str
    environment: NotRequired[Optional[str]]
    successful_files: List[str]
    failed_files: List[str]
    error: NotRequired[Optional[str]]
    duration_ms: float

class Metadata(TypedDict):
    """Type definition for the metadata structure."""
    last_operation: Optional[OperationData]
    history: List[OperationData]

class MetadataHandler:
    """Handles metadata storage and retrieval for sync operations.
    
    This class manages a JSON file that stores the last 5 sync operations
    and maintains a record of the most recent operation with detailed information.
    """
    
    def __init__(self, base_dir: Path, max_history: int = 5) -> None:
        """Initialize the metadata handler.
        
        Args:
            base_dir: Base directory where metadata will be stored
            max_history: Maximum number of operations to keep in history
        """
        self.base_dir = base_dir
        self.metadata_file = base_dir / ".sync_metadata.json"
        self.max_history = max_history
        self._ensure_metadata_file()
    
    def _ensure_metadata_file(self) -> None:
        """Ensure the metadata file exists with proper structure."""
        if not self.metadata_file.exists():
            initial_data: Metadata = {
                "last_operation": None,
                "history": []
            }
            self._write_metadata(initial_data)
    
    def _read_metadata(self) -> Metadata:
        """Read the current metadata from file."""
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading metadata file: {e}")
            return {"last_operation": None, "history": []}
    
    def _write_metadata(self, data: Metadata) -> None:
        """Write metadata to file."""
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing metadata file: {e}")
    
    def log_operation(
        self,
        operation_type: str,
        path: str,
        duration_ms: float,
        environment: Optional[str] = None,
        successful_files: Optional[List[str]] = None,
        failed_files: Optional[List[str]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log a sync operation.
        
        Args:
            operation_type: Type of operation (e.g., "pull", "push")
            path: The path that was synced
            duration_ms: Duration of the operation in milliseconds
            environment: Optional environment name
            successful_files: List of successfully processed files
            failed_files: List of files that failed to process
            error: Any error message if the operation failed
        """
        current_time = datetime.now().isoformat()
        
        operation_data: OperationData = {
            "timestamp": current_time,
            "operation_type": operation_type,
            "path": path,
            "environment": environment,
            "successful_files": successful_files or [],
            "failed_files": failed_files or [],
            "error": error,
            "duration_ms": duration_ms
        }
        
        metadata = self._read_metadata()
        
        # Update last operation
        metadata["last_operation"] = operation_data
        
        # Update history
        metadata["history"].insert(0, operation_data)
        metadata["history"] = metadata["history"][:self.max_history]
        
        self._write_metadata(metadata)
    
    def get_last_operation(self) -> Optional[OperationData]:
        """Get the most recent operation details."""
        metadata = self._read_metadata()
        return metadata.get("last_operation")
    
    def get_history(self) -> List[OperationData]:
        """Get the operation history."""
        metadata = self._read_metadata()
        return metadata.get("history", []) 