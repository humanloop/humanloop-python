import logging
import types
from typing import TypeVar, Union
import typing

from humanloop.context import get_trace_id
from humanloop.eval_utils.run import HumanloopUtilityError

from humanloop.prompts.client import PromptsClient
from humanloop.types.create_evaluator_log_response import CreateEvaluatorLogResponse
from humanloop.types.create_flow_log_response import CreateFlowLogResponse
from humanloop.types.create_prompt_log_response import CreatePromptLogResponse
from humanloop.types.create_tool_log_response import CreateToolLogResponse
from humanloop.types.prompt_call_response import PromptCallResponse

logger = logging.getLogger("humanloop.sdk")


CLIENT_TYPE = TypeVar("CLIENT_TYPE")


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
        if trace_id is not None:
            if "trace_parent_id" in kwargs:
                # TODO: revisit
                logger.warning("Overriding trace_parent_id argument")
            kwargs = {
                **kwargs,
                "trace_parent_id": trace_id,
            }
        try:
            response = self._log(**kwargs)
        except Exception as e:
            # TODO handle
            # TODO: Bug found in backend: not specifying a model 400s but creates a File
            raise HumanloopUtilityError(message=str(e)) from e

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
                # TODO: revisit
                logger.warning("Overriding trace_parent_id argument")
            kwargs = {
                **kwargs,
                "trace_parent_id": trace_id,
            }

        try:
            response = self._call(**kwargs)
            response = typing.cast(PromptCallResponse, response)
        except Exception as e:
            # TODO handle
            # TODO: Bug found in backend: not specifying a model 400s but creates a File
            raise HumanloopUtilityError(message=str(e)) from e

        return response

    # Replace the original log method with the overloaded one
    client.call = types.MethodType(_overload_call, client)
    return client
