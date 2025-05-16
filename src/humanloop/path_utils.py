from pathlib import Path
from typing import Optional

from humanloop.error import HumanloopRuntimeError


def validate_no_extensions(path: str, allowed_extensions: Optional[list[str]] = None) -> str:
    """Validates that path has no extensions (or only allowed ones) and returns path without extension."""
    if "." in path:
        parts = path.rsplit(".", 1)
        extension = parts[1] if len(parts) > 1 else ""
        path_without_extension = parts[0]
        
        if allowed_extensions and extension in allowed_extensions:
            return path
        
        raise HumanloopRuntimeError(
            f"Path '{path}' includes a file extension which is not supported in this context. "
            f"Use the format without extensions: '{path_without_extension}'."
        )
    return path

def validate_no_slashes(path: str) -> str:
    """Validates that path has no leading/trailing slashes."""
    if path != path.strip("/"):
        raise HumanloopRuntimeError(
            f"Invalid path: '{path}'. Path should not contain leading/trailing slashes. "
            f"Valid example: 'path/to/resource'"
        )
    return path

def validate_not_absolute(path: str, base_dir: Optional[str] = None) -> str:
    """Validates that path is not absolute."""
    if Path(path).is_absolute():
        message = f"Absolute paths are not supported: '{path}'."
        if base_dir:
            message += f" Paths should be relative to '{base_dir}'."
        raise HumanloopRuntimeError(message)
    return path