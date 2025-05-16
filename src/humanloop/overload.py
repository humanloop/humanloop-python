import inspect
import logging
import types
from typing import Any, Callable, Dict, Optional, TypeVar, Union

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


T = TypeVar("T", bound=Union[PromptsClient, AgentsClient, ToolsClient, FlowsClient, DatasetsClient, EvaluatorsClient])


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
    else:
        raise ValueError(f"Unsupported client type: {type(client)}")


def _handle_tracing_context(kwargs: Dict[str, Any], client: T) -> Dict[str, Any]:
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
    client: T,
    sync_client: Optional[SyncClient],
    use_local_files: bool,
) -> Dict[str, Any]:
    """Handle local file loading if enabled."""
    if not use_local_files or "path" not in kwargs or sync_client is None:
        return kwargs

    if "id" in kwargs:
        raise HumanloopRuntimeError("Can only specify one of `id` or `path`")

    # Check if version_id or environment is specified
    use_remote = any(["version_id" in kwargs, "environment" in kwargs])
    normalized_path = sync_client._normalize_path(kwargs["path"])

    if use_remote:
        raise HumanloopRuntimeError(
            f"Cannot use local file for `{normalized_path}` as version_id or environment was specified. "
            "Please either remove version_id/environment to use local files, or set use_local_files=False to use remote files."
        )

    file_type = _get_file_type_from_client(client)
    if file_type not in SyncClient.SERIALIZABLE_FILE_TYPES:
        raise HumanloopRuntimeError(f"Local files are not supported for `{file_type}` files.")

    # If file_type is already specified in kwargs, it means user provided a PromptKernelRequestParams object
    if file_type in kwargs and not isinstance(kwargs[file_type], str):
        logger.warning(
            f"Ignoring local file for `{normalized_path}` as {file_type} parameters were directly provided. "
            "Using provided parameters instead."
        )
        return kwargs

    try:
        file_content = sync_client.get_file_content(normalized_path, file_type)  # type: ignore[arg-type] # file_type was checked above
        kwargs[file_type] = file_content
    except HumanloopRuntimeError as e:
        raise HumanloopRuntimeError(f"Failed to use local file for `{normalized_path}`: {str(e)}")

    return kwargs


def _handle_evaluation_context(kwargs: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[Callable[[str], None]]]:
    """Handle evaluation context for logging."""
    evaluation_context = get_evaluation_context()
    if evaluation_context is not None:
        return evaluation_context.log_args_with_context(path=kwargs.get("path"), log_args=kwargs)
    return kwargs, None


def _overload_log(self: T, sync_client: Optional[SyncClient], use_local_files: bool, **kwargs) -> LogResponseType:
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
        if _get_file_type_from_client(self) in ["prompt", "agent"]:
            if sync_client is None:
                logger.error("sync_client is None but client has log method and use_local_files=%s", use_local_files)
                raise HumanloopRuntimeError("sync_client is required for clients that support local file operations")
            kwargs = _handle_local_files(kwargs, self, sync_client, use_local_files)

        kwargs, eval_callback = _handle_evaluation_context(kwargs)
        response = self._log(**kwargs)  # type: ignore[union-attr] # Use stored original method
        if eval_callback is not None:
            eval_callback(response.id)
        return response
    except HumanloopRuntimeError:
        # Re-raise HumanloopRuntimeError without wrapping to preserve the message
        raise
    except Exception as e:
        # Only wrap non-HumanloopRuntimeError exceptions
        raise HumanloopRuntimeError from e


def _overload_call(self: T, sync_client: Optional[SyncClient], use_local_files: bool, **kwargs) -> CallResponseType:
    try:
        kwargs = _handle_tracing_context(kwargs, self)
        kwargs = _handle_local_files(kwargs, self, sync_client, use_local_files)
        return self._call(**kwargs)  # type: ignore[union-attr] # Use stored original method
    except HumanloopRuntimeError:
        # Re-raise HumanloopRuntimeError without wrapping to preserve the message
        raise
    except Exception as e:
        # Only wrap non-HumanloopRuntimeError exceptions
        raise HumanloopRuntimeError from e


def overload_client(
    client: T,
    sync_client: Optional[SyncClient] = None,
    use_local_files: bool = False,
) -> T:
    """Overloads client methods to add tracing, local file handling, and evaluation context."""
    # Store original log method as _log for all clients. Used in flow decorator
    if hasattr(client, "log") and not hasattr(client, "_log"):
        # Store original method with type ignore
        client._log = client.log  # type: ignore

        # Create a closure to capture sync_client and use_local_files
        def log_wrapper(self: T, **kwargs) -> LogResponseType:
            return _overload_log(self, sync_client, use_local_files, **kwargs)

        # Replace the log method with type ignore
        client.log = types.MethodType(log_wrapper, client)  # type: ignore

    # Overload call method for Prompt and Agent clients
    if _get_file_type_from_client(client) in ["prompt", "agent"]:
        if sync_client is None and use_local_files:
            logger.error("sync_client is None but client has call method and use_local_files=%s", use_local_files)
            raise HumanloopRuntimeError("sync_client is required for clients that support call operations")
        if hasattr(client, "call") and not hasattr(client, "_call"):
            # Store original method with type ignore
            client._call = client.call  # type: ignore

            # Create a closure to capture sync_client and use_local_files
            def call_wrapper(self: T, **kwargs) -> CallResponseType:
                return _overload_call(self, sync_client, use_local_files, **kwargs)

            # Replace the call method with type ignore
            client.call = types.MethodType(call_wrapper, client)  # type: ignore

    return client
