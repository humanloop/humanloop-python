from pathlib import Path


def normalize_path(path: str, strip_extension: bool = False) -> str:
    """Normalize a path to the standard API format: path/to/resource, where resource can be a file or a directory."""
    # Handle backslashes for Windows paths before passing to PurePosixPath
    # This is needed because some backslash sequences are treated as escape chars
    path = path.replace("\\", "/")

    # Use PurePosixPath to normalize the path (handles consecutive slashes)
    path_obj = Path(path)

    # Strip extension if requested
    if strip_extension:
        path_obj = path_obj.with_suffix("")

    # Convert to string and remove any leading/trailing slashes
    # We use the path as a string and not as_posix() since we've already normalized separators
    return str(path_obj).strip("/")
