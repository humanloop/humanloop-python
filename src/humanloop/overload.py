import inspect
import logging
import types
from typing import TypeVar, Union, Literal
from pathlib import Path
from humanloop.context import (
    get_decorator_context,
    get_evaluation_context,
    get_trace_id,
)
from humanloop.evals.run import HumanloopRuntimeError

from humanloop.evaluators.client import EvaluatorsClient
from humanloop.flows.client import FlowsClient
from humanloop.prompts.client import PromptsClient
from humanloop.agents.client import AgentsClient
from humanloop.tools.client import ToolsClient
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
                # Re-raising as HumanloopDecoratorError so the decorators don't catch it
                raise HumanloopRuntimeError from e
            if eval_callback is not None:
                eval_callback(response.id)
        else:
            try:
                response = self._log(**kwargs)
            except Exception as e:
                # Re-raising as HumanloopDecoratorError so the decorators don't catch it
                raise HumanloopRuntimeError from e

        return response

    # Replace the original log method with the overloaded one
    client.log = types.MethodType(_overload_log, client)  # type: ignore [assignment]
    # Return the client with the overloaded log method
    logger.debug("Overloaded the .call method of %s", client)
    return client


def overload_call(client: PromptsClient) -> PromptsClient:
    client._call = client.call  # type: ignore [attr-defined]

    def _overload_call(self, **kwargs) -> PromptCallResponse:
        # None if not logging inside a decorator
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
            logger.info(f"Calling inner overload")
            response = self._call(**kwargs)
        except Exception as e:
            # Re-raising as HumanloopDecoratorError so the decorators don't catch it
            raise HumanloopRuntimeError from e

        return response

    # Replace the original log method with the overloaded one
    client.call = types.MethodType(_overload_call, client)  # type: ignore [assignment]
    return client

def overload_call_with_local_files(
    client: Union[PromptsClient, AgentsClient], 
    use_local_files: bool,
) -> Union[PromptsClient, AgentsClient]:
    """Overload call to handle local files when use_local_files is True.
    
    Args:
        client: The client to overload (PromptsClient or AgentsClient)
        use_local_files: Whether to use local files
        file_type: Type of file ("prompt" or "agent")
    """
    original_call = client._call if hasattr(client, '_call') else client.call
    # get file type from client type
    file_type: FileType
    if isinstance(client, PromptsClient):   
        file_type = "prompt"    
    elif isinstance(client, AgentsClient):
        file_type = "agent"
    else:
        raise ValueError(f"Unsupported client type: {type(client)}")

    def _overload_call(self, **kwargs) -> PromptCallResponse:
        if use_local_files and "path" in kwargs:
            try:
                # Construct path to local file
                local_path = Path("humanloop") / kwargs["path"]
                # Add appropriate extension
                local_path = local_path.parent / f"{local_path.stem}.{file_type}"
                
                if local_path.exists():
                    # Read the file content
                    with open(local_path) as f:
                        file_content = f.read()
                    
                    kwargs[file_type] = file_content  # "prompt" or "agent" # TODO: raise warning if kernel passed in
                    
                    logger.debug(f"Using local file content from {local_path}")
                else:
                    logger.warning(f"Local file not found: {local_path}, falling back to API")
            except Exception as e:
                logger.error(f"Error reading local file: {e}, falling back to API")

        return original_call(**kwargs)

    client.call = types.MethodType(_overload_call, client)
    return client