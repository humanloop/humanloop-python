from collections import defaultdict
import inspect
import logging
import types
import warnings
from typing import TypeVar, Union
from pathlib import Path
from humanloop.context import (
    get_decorator_context,
    get_evaluation_context,
    get_trace_id,
)
from humanloop.error import HumanloopRuntimeError

from humanloop.evaluators.client import EvaluatorsClient
from humanloop.flows.client import FlowsClient
from humanloop.prompts.client import PromptsClient
from humanloop.agents.client import AgentsClient
from humanloop.tools.client import ToolsClient
from humanloop.sync.sync_client import SyncClient
from humanloop.types import FileType
from humanloop.types.create_evaluator_log_response import CreateEvaluatorLogResponse
from humanloop.types.create_flow_log_response import CreateFlowLogResponse
from humanloop.types.create_prompt_log_response import CreatePromptLogResponse
from humanloop.types.create_tool_log_response import CreateToolLogResponse
from humanloop.types.prompt_call_response import PromptCallResponse

logger = logging.getLogger("humanloop.sdk")


CLIENT_TYPE = TypeVar("CLIENT_TYPE", PromptsClient, FlowsClient, EvaluatorsClient, ToolsClient)


def overload_log(client: CLIENT_TYPE) -> CLIENT_TYPE:
    """
    Wrap the `log` method of the provided Humanloop client to use EVALUATION_CONTEXT.

    This makes the overloaded log actions be aware of whether the created Log is
    part of an Evaluation (e.g. one started by eval_utils.run_eval).
    """
    # Copy the original log method in a hidden attribute
    client._log = client.log  # type: ignore [attr-defined]

    def _overload_log(
        # It's safe to only consider kwargs since the original
        # log method bans positional arguments
        self,
        **kwargs,
    ) -> Union[
        CreatePromptLogResponse,
        CreateToolLogResponse,
        CreateFlowLogResponse,
        CreateEvaluatorLogResponse,
    ]:
        trace_id = get_trace_id()
        if trace_id is not None and type(client) is FlowsClient:
            context = get_decorator_context()
            if context is None:
                raise HumanloopRuntimeError("Internal error: trace_id context is set outside a decorator context.")
            raise HumanloopRuntimeError(
                f"Using `flows.log()` is not allowed: Flow decorator "
                f"for File {context.path} manages the tracing and trace completion."
            )
        if trace_id is not None:
            if "trace_parent_id" in kwargs:
                logger.warning(
                    "Ignoring trace_parent_id argument at line %d: the Flow decorator manages tracing.",
                    inspect.currentframe().f_lineno,  # type: ignore [union-attr]
                )
            kwargs = {
                **kwargs,
                "trace_parent_id": trace_id,
            }
        evaluation_context = get_evaluation_context()
        if evaluation_context is not None:
            kwargs_eval, eval_callback = evaluation_context.log_args_with_context(
                path=kwargs.get("path"), log_args=kwargs
            )
            try:
                response = self._log(**kwargs_eval)
            except Exception as e:
                # Re-raising as HumanloopRuntimeError so the decorators don't catch it
                raise HumanloopRuntimeError from e
            if eval_callback is not None:
                eval_callback(response.id)
        else:
            try:
                response = self._log(**kwargs)
            except Exception as e:
                # Re-raising as HumanloopRuntimeError so the decorators don't catch it
                raise HumanloopRuntimeError from e

        return response

    # Replace the original log method with the overloaded one
    client.log = types.MethodType(_overload_log, client)  # type: ignore [assignment]
    # Return the client with the overloaded log method
    logger.debug("Overloaded the .call method of %s", client)
    return client


def overload_call(client: PromptsClient) -> PromptsClient:
    if not hasattr(client, "_overloads"):
        client._overloads = defaultdict(list)  # type: ignore [attr-defined]
    if len(client._overloads["call"]) == 0:  # type: ignore [attr-defined]
        client._overloads["call"].append(client.call)  # type: ignore [attr-defined]

    def _overload_call(self, **kwargs) -> PromptCallResponse:
        # trace_id is None if logging outside a decorator
        trace_id = get_trace_id()
        if trace_id is not None:
            if "trace_parent_id" in kwargs:
                logger.warning(
                    "Ignoring trace_parent_id argument at line %d: the Flow decorator manages tracing.",
                    inspect.currentframe().f_lineno,  # type: ignore [union-attr]
                )
            kwargs = {
                **kwargs,
                "trace_parent_id": trace_id,
            }

        try:
            response = client._overloads["call"][0](**kwargs)  # type: ignore [attr-defined]
        except Exception as e:
            # Re-raising as HumanloopRuntimeError so the decorators don't catch it
            raise HumanloopRuntimeError from e

        return response

    client.call = types.MethodType(_overload_call, client)  # type: ignore [assignment]
    return client


def _get_file_type_from_client(client: Union[PromptsClient, AgentsClient]) -> FileType:
    """Get the file type based on the client type."""
    if isinstance(client, PromptsClient):
        return "prompt"
    elif isinstance(client, AgentsClient):
        return "agent"
    else:
        raise ValueError(f"Unsupported client type: {type(client)}")


def overload_with_local_files(
    client: Union[PromptsClient, AgentsClient],
    sync_client: SyncClient,
    use_local_files: bool,
) -> Union[PromptsClient, AgentsClient]:
    """Overload call and log methods to handle local files when use_local_files is True.

    When use_local_files is True, the following prioritization strategy is used:
    1. Direct Parameters: If {file_type} parameters are provided directly (as a PromptKernelRequestParams or AgentKernelRequestParams object),
       these take precedence and the local file is ignored.
    2. Version/Environment: If version_id or environment is specified, the remote version is used instead
       of the local file.
    3. Local File: If neither of the above are specified, attempts to use the local file at the given path.

    For example, with a prompt client:
    - If prompt={model: "gpt-4", ...} is provided, uses those parameters directly
    - If version_id="123" is provided, uses that remote version
    - Otherwise, tries to load from the local file at the given path

    Args:
        client: The client to overload (PromptsClient or AgentsClient)
        sync_client: The sync client used for file operations
        use_local_files: Whether to enable local file handling

    Returns:
        The client with overloaded methods

    Raises:
        HumanloopRuntimeError: If use_local_files is True and local file cannot be accessed
    """
    if not hasattr(client, "_overloads"):
        client._overloads = defaultdict(list)  # type: ignore [union-attr]
    # If the method has been overloaded, don't re-add the method
    if isinstance(client, PromptsClient):
        if len(client._overloads["call"]) == 1:  # type: ignore [attr-defined]
            client._overloads["call"].append(client.call)  # type: ignore [attr-defined]
        else:
            raise RuntimeError(f"Unexpected overload order of operations for {client}.call")
    elif isinstance(client, AgentsClient):
        if len(client._overloads["call"]) == 0:
            client._overloads["call"].append(client.call)
        else:
            raise RuntimeError(f"Unexpected overload order of operations for {client}.call")
    else:
        raise NotImplementedError(f"Unsupported client type: {type(client)}")
    if len(client._overloads["log"]) == 0:
        client._overloads["log"].append(client.log)
    else:
        raise RuntimeError(f"Unexpected overload order of operations for {client}.log")

    file_type = _get_file_type_from_client(client)

    def _overload(self, function_name: str, **kwargs) -> PromptCallResponse:
        if "id" in kwargs and "path" in kwargs:
            raise HumanloopRuntimeError(
                "Can only specify one of `id` or `path` when "
                f"{'logging' if function_name == 'log' else 'calling'} a {file_type}"
            )
        # Handle local files if enabled
        if use_local_files and "path" in kwargs:
            # Check if version_id or environment is specified
            use_remote = any(["version_id" in kwargs, "environment" in kwargs])
            normalized_path = sync_client._normalize_path(kwargs["path"])

            if use_remote:
                raise HumanloopRuntimeError(
                    f"Cannot use local file for `{normalized_path}` as version_id or environment was specified. "
                    "Please either remove version_id/environment to use local files, or set use_local_files=False to use remote files."
                )
            else:
                # Only use local file if no version info is specified
                try:
                    # If file_type is already specified in kwargs, it means user provided a PromptKernelRequestParams object
                    if file_type in kwargs and not isinstance(kwargs[file_type], str):
                        logger.warning(
                            f"Ignoring local file for `{normalized_path}` as {file_type} parameters were directly provided. "
                            "Using provided parameters instead."
                        )
                    else:
                        file_content = sync_client.get_file_content(normalized_path, file_type)
                        kwargs[file_type] = file_content
                except HumanloopRuntimeError as e:
                    # Re-raise with more context
                    raise HumanloopRuntimeError(f"Failed to use local file for `{normalized_path}`: {str(e)}")

        try:
            if function_name == "call":
                return client._overloads["call"][1](**kwargs)  # type: ignore [attr-defined, union-attr]
            elif function_name == "log":
                return client._overloads["log"][0](**kwargs)  # type: ignore [attr-defined, union-attr]
            else:
                raise ValueError(f"Unsupported function name: {function_name}")
        except Exception as e:
            # Re-raising as HumanloopRuntimeError so the decorators don't catch it
            raise HumanloopRuntimeError from e

    def _overload_call(self, **kwargs) -> PromptCallResponse:
        return _overload(self, "call", **kwargs)

    def _overload_log(self, **kwargs) -> PromptCallResponse:
        return _overload(self, "log", **kwargs)

    client.call = types.MethodType(_overload_call, client)  # type: ignore [assignment]
    client.log = types.MethodType(_overload_log, client)  # type: ignore [assignment]
    return client
