import inspect
import logging
import types
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from humanloop import path_utils
from humanloop.agents.client import AgentsClient
from humanloop.context import (
    get_decorator_context,
    get_evaluation_context,
    get_trace_id,
)
from humanloop.datasets.client import DatasetsClient
from humanloop.error import HumanloopRuntimeError
from humanloop.evaluators.client import EvaluatorsClient
from humanloop.flows.client import FlowsClient
from humanloop.prompts.client import PromptsClient
from humanloop.sync.sync_client import SyncClient
from humanloop.tools.client import ToolsClient
from humanloop.types import FileType
from humanloop.types.agent_call_response import AgentCallResponse
from humanloop.types.create_evaluator_log_response import CreateEvaluatorLogResponse
from humanloop.types.create_flow_log_response import CreateFlowLogResponse
from humanloop.types.create_prompt_log_response import CreatePromptLogResponse
from humanloop.types.create_tool_log_response import CreateToolLogResponse
from humanloop.types.prompt_call_response import PromptCallResponse

logger = logging.getLogger("humanloop.sdk")

LogResponseType = Union[
    CreatePromptLogResponse,
    CreateToolLogResponse,
    CreateFlowLogResponse,
    CreateEvaluatorLogResponse,
]

CallResponseType = Union[
    PromptCallResponse,
    AgentCallResponse,
]


def _get_file_type_from_client(
    client: Union[PromptsClient, AgentsClient, ToolsClient, FlowsClient, DatasetsClient, EvaluatorsClient],
) -> FileType:
    """Get the file type based on the client type."""
    if isinstance(client, PromptsClient):
        return "prompt"
    elif isinstance(client, AgentsClient):
        return "agent"
    elif isinstance(client, ToolsClient):
        return "tool"
    elif isinstance(client, FlowsClient):
        return "flow"
    elif isinstance(client, DatasetsClient):
        return "dataset"
    elif isinstance(client, EvaluatorsClient):
        return "evaluator"

    raise ValueError(f"Unsupported client type: {type(client)}")


def _handle_tracing_context(kwargs: Dict[str, Any], client: Any) -> Dict[str, Any]:
    """Handle tracing context for both log and call methods."""
    trace_id = get_trace_id()
    if trace_id is not None:
        if "flow" in str(type(client).__name__).lower():
            context = get_decorator_context()
            if context is None:
                raise HumanloopRuntimeError("Internal error: trace_id context is set outside a decorator context.")
            raise HumanloopRuntimeError(
                f"Using `flows.log()` is not allowed: Flow decorator "
                f"for File {context.path} manages the tracing and trace completion."
            )

        if "trace_parent_id" in kwargs:
            logger.warning(
                "Ignoring trace_parent_id argument at line %d: the Flow decorator manages tracing.",
                inspect.currentframe().f_lineno,  # type: ignore[union-attr]
            )
        kwargs = {
            **kwargs,
            "trace_parent_id": trace_id,
        }
    return kwargs


def _handle_local_files(
    kwargs: Dict[str, Any],
    client: Any,
    sync_client: SyncClient,
) -> Dict[str, Any]:
    """Handle local file loading."""
    if "id" in kwargs:
        raise HumanloopRuntimeError("Can only specify one of `id` or `path`")

    path = kwargs["path"]

    # First check for path format issues (absolute paths or leading/trailing slashes)
    normalized_path = path.strip("/")
    if Path(path).is_absolute() or path != normalized_path:
        raise HumanloopRuntimeError(
            f"Path '{path}' format is invalid. "
            f"Paths must follow the standard API format 'path/to/resource' without leading or trailing slashes. "
            f"Please use '{normalized_path}' instead."
        )

    # Then check for file extensions
    if sync_client.is_file(path):
        try:
            parts = path.rsplit(".", 1)
            path_without_extension = parts[0] if len(parts) > 0 else path
        except Exception:
            path_without_extension = path

        raise HumanloopRuntimeError(
            f"Path '{path}' includes a file extension which is not supported in API calls. "
            f"When referencing files via the path parameter, use the format without extensions: '{path_without_extension}'. "
            f"Note: File extensions are only used when pulling specific files via the CLI."
        )

    # Check if version_id or environment is specified
    use_remote = any(["version_id" in kwargs, "environment" in kwargs])

    if use_remote:
        raise HumanloopRuntimeError(
            f"Cannot use local file for `{path}` as version_id or environment was specified. "
            "Please either remove version_id/environment to use local files, or set use_local_files=False to use remote files."
        )

    file_type = _get_file_type_from_client(client)
    if file_type not in SyncClient.SERIALIZABLE_FILE_TYPES:
        raise HumanloopRuntimeError(f"Local files are not supported for `{file_type.capitalize()}` files: '{path}'.")

    # If file_type is already specified in kwargs (prompt or agent), it means user provided a Prompt- or AgentKernelRequestParams object
    if file_type in kwargs and not isinstance(kwargs[file_type], str):
        logger.warning(
            f"Ignoring local file for `{path}` as {file_type} parameters were directly provided. "
            "Using provided parameters instead."
        )
        return kwargs

    try:
        file_content = sync_client.get_file_content(path, file_type)  # type: ignore[arg-type] # file_type was checked above
        kwargs[file_type] = file_content

        return kwargs
    except HumanloopRuntimeError as e:
        raise HumanloopRuntimeError(f"Failed to use local file for `{path}`: {str(e)}")


def _handle_evaluation_context(kwargs: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[Callable[[str], None]]]:
    """Handle evaluation context for logging."""
    evaluation_context = get_evaluation_context()
    if evaluation_context is not None:
        return evaluation_context.log_args_with_context(path=kwargs.get("path"), log_args=kwargs)
    return kwargs, None


def _overload_log(self: Any, sync_client: Optional[SyncClient], use_local_files: bool, **kwargs) -> LogResponseType:
    try:
        # Special handling for flows - prevent direct log usage
        if type(self) is FlowsClient and get_trace_id() is not None:
            context = get_decorator_context()
            if context is None:
                raise HumanloopRuntimeError("Internal error: trace_id context is set outside a decorator context.")
            raise HumanloopRuntimeError(
                f"Using `flows.log()` is not allowed: Flow decorator "
                f"for File {context.path} manages the tracing and trace completion."
            )

        kwargs = _handle_tracing_context(kwargs, self)

        # Handle local files for Prompts and Agents clients
        if use_local_files and _get_file_type_from_client(self) in SyncClient.SERIALIZABLE_FILE_TYPES:
            if sync_client is None:
                logger.error("sync_client is None but client has log method and use_local_files=%s", use_local_files)
                raise HumanloopRuntimeError("sync_client is required for clients that support local file operations")
            kwargs = _handle_local_files(kwargs, self, sync_client)

        kwargs, eval_callback = _handle_evaluation_context(kwargs)
        response = self._log(**kwargs)  # Use stored original method
        if eval_callback is not None:
            eval_callback(response.id)
        return response
    except HumanloopRuntimeError:
        # Re-raise HumanloopRuntimeError without wrapping to preserve the message
        raise
    except Exception as e:
        # Only wrap non-HumanloopRuntimeError exceptions
        raise HumanloopRuntimeError from e


def _overload_call(self: Any, sync_client: Optional[SyncClient], use_local_files: bool, **kwargs) -> CallResponseType:
    try:
        kwargs = _handle_tracing_context(kwargs, self)
        if use_local_files and _get_file_type_from_client(self) in SyncClient.SERIALIZABLE_FILE_TYPES:
            if sync_client is None:
                logger.error("sync_client is None but client has call method and use_local_files=%s", use_local_files)
                raise HumanloopRuntimeError("sync_client is required for clients that support call operations")
            kwargs = _handle_local_files(kwargs, self, sync_client)
        return self._call(**kwargs)  # Use stored original method
    except HumanloopRuntimeError:
        # Re-raise HumanloopRuntimeError without wrapping to preserve the message
        raise
    except Exception as e:
        # Only wrap non-HumanloopRuntimeError exceptions
        raise HumanloopRuntimeError from e


def overload_client(
    client: Any,
    sync_client: Optional[SyncClient] = None,
    use_local_files: bool = False,
) -> Any:
    """Overloads client methods to add tracing, local file handling, and evaluation context."""
    # Store original log method as _log for all clients. Used in flow decorator
    if hasattr(client, "log") and not hasattr(client, "_log"):
        client._log = client.log  # type: ignore[attr-defined]

        # Create a closure to capture sync_client and use_local_files
        def log_wrapper(self: Any, **kwargs) -> LogResponseType:
            return _overload_log(self, sync_client, use_local_files, **kwargs)

        client.log = types.MethodType(log_wrapper, client)

    # Overload call method for Prompt and Agent clients
    if _get_file_type_from_client(client) in SyncClient.SERIALIZABLE_FILE_TYPES:
        if sync_client is None and use_local_files:
            logger.error("sync_client is None but client has call method and use_local_files=%s", use_local_files)
            raise HumanloopRuntimeError("sync_client is required for clients that support call operations")
        if hasattr(client, "call") and not hasattr(client, "_call"):
            client._call = client.call  # type: ignore[attr-defined]

            # Create a closure to capture sync_client and use_local_files
            def call_wrapper(self: Any, **kwargs) -> CallResponseType:
                return _overload_call(self, sync_client, use_local_files, **kwargs)

            client.call = types.MethodType(call_wrapper, client)

    return client
