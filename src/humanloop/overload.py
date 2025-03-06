import inspect
import logging
import types
from typing import TypeVar, Union
import typing

from humanloop.context import get_decorator_context, get_trace_id
from humanloop.eval_utils.run import HumanloopDecoratorError

from humanloop.evaluators.client import EvaluatorsClient
from humanloop.flows.client import FlowsClient
from humanloop.prompts.client import PromptsClient
from humanloop.tools.client import ToolsClient
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
    client._log = client.log

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
                raise HumanloopDecoratorError("Internal error: trace_id context is set outside a decorator context.")
            raise HumanloopDecoratorError(
                f"Using flows.log() in this context is not allowed at line {inspect.currentframe().f_lineno}: "
                f"Flow decorator for File {context.path} manages the tracing and trace completion."
            )
        if trace_id is not None:
            if "trace_parent_id" in kwargs:
                logger.warning(
                    "Ignoring trace_parent_id argument at line %d: the Flow decorator manages tracing.",
                    inspect.currentframe().f_lineno,
                )
            kwargs = {
                **kwargs,
                "trace_parent_id": trace_id,
            }
        try:
            response = self._log(**kwargs)
        except Exception as e:
            # Re-raising as HumanloopDecoratorError so the decorators don't catch it
            raise HumanloopDecoratorError from e

        return response

    # Replace the original log method with the overloaded one
    client.log = types.MethodType(_overload_log, client)
    # Return the client with the overloaded log method
    logger.debug("Overloaded the .call method of %s", client)
    return client


def overload_call(client: PromptsClient) -> PromptsClient:
    client._call = client.call

    def _overload_call(self, **kwargs) -> PromptCallResponse:
        # None if not logging inside a decorator
        trace_id = get_trace_id()
        if trace_id is not None:
            if "trace_parent_id" in kwargs:
                logger.warning(
                    "Ignoring trace_parent_id argument at line %d: the Flow decorator manages tracing.",
                    inspect.currentframe().f_lineno,
                )
            kwargs = {
                **kwargs,
                "trace_parent_id": trace_id,
            }

        try:
            response = self._call(**kwargs)
            response = typing.cast(PromptCallResponse, response)
        except Exception as e:
            # Re-raising as HumanloopDecoratorError so the decorators don't catch it
            raise HumanloopDecoratorError from e

        return response

    # Replace the original log method with the overloaded one
    client.call = types.MethodType(_overload_call, client)
    return client
