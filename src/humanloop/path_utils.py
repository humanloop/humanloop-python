from pathlib import Path


def normalize_path(path: str, strip_extension: bool = False) -> str:
    """Normalize a path to the standard Humanloop API format.

    This function is primarily used when interacting with the Humanloop API to ensure paths
    follow the standard format: 'path/to/resource' without leading/trailing slashes.
    It's used when pulling files from Humanloop to local filesystem (see FileSyncer.pull)

    The function:
    - Converts Windows backslashes to forward slashes
    - Normalizes consecutive slashes
    - Optionally strips file extensions (e.g. .prompt, .agent)
    - Removes leading/trailing slashes to match API conventions

    Leading/trailing slashes are stripped because the Humanloop API expects paths in the
    format 'path/to/resource' without them. This is consistent with how the API stores
    and references files, and ensures paths work correctly in both API calls and local
    filesystem operations.

    Args:
        path: The path to normalize. Can be a Windows or Unix-style path.
        strip_extension: If True, removes the file extension (e.g. .prompt, .agent)

    Returns:
        Normalized path string in the format 'path/to/resource'

    Examples:
        >>> normalize_path("path/to/file.prompt")
        'path/to/file.prompt'
        >>> normalize_path("path/to/file.prompt", strip_extension=True)
        'path/to/file'
        >>> normalize_path("\\windows\\style\\path.prompt")
        'windows/style/path.prompt'
        >>> normalize_path("/leading/slash/path/")
        'leading/slash/path'
        >>> normalize_path("multiple//slashes//path")
        'multiple/slashes/path'
    """
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
