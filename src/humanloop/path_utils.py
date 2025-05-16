from pathlib import Path


def normalize_path(path: str, strip_extension: bool = False) -> str:
    """Normalize a path to the standard API format: path/to/resource, where resource can be a file or a directory."""
    # Remove leading/trailing slashes
    path_obj = Path(path.strip("/"))

    # Handle extension
    if strip_extension:
        path_obj = path_obj.with_suffix("")

    # Convert path to string with forward slashes, regardless of OS
    # This ensures consistent path format (e.g., "path/to/resource") across Windows/Unix
    return path_obj.as_posix()
