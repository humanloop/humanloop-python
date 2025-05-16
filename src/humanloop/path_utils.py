from pathlib import Path

from humanloop.error import HumanloopRuntimeError


def normalize_path(path: str, strip_extension: bool = False) -> str:
    """Normalize a path to the standard API format: path/to/resource, where resource can be a file or a directory."""
    # Remove leading/trailing slashes
    path_obj = Path(path.strip("/"))

    # Check if path is absolute
    if path_obj.is_absolute():
        raise HumanloopRuntimeError("Absolute paths are not supported. Ensure the ")

    # Handle extension
    if strip_extension:
        normalized = str(path_obj.with_suffix(""))
    else:
        normalized = str(path_obj)

    # Normalize separators
    return "/".join(part for part in normalized.replace("\\", "/").split("/") if part)
