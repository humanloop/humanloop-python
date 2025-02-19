import logging
import threading
import types
from typing import TypeVar, Union
import typing

from humanloop.context_variables import (
    get_evaluation_context,
    get_prompt_utility_context,
    in_prompt_utility_context,
    log_belongs_to_evaluated_file,
)
from humanloop.eval_utils.run import HumanloopUtilityError
from humanloop.flows.client import FlowsClient
from humanloop.otel.constants import HUMANLOOP_INTERCEPTED_HL_CALL_RESPONSE, HUMANLOOP_INTERCEPTED_HL_CALL_SPAN_NAME
from humanloop.otel.helpers import write_to_opentelemetry_span
from humanloop.prompts.client import PromptsClient
from humanloop.types.create_evaluator_log_response import CreateEvaluatorLogResponse
from humanloop.types.create_flow_log_response import CreateFlowLogResponse
from humanloop.types.create_prompt_log_response import CreatePromptLogResponse
from humanloop.types.create_tool_log_response import CreateToolLogResponse
from humanloop.types.prompt_call_response import PromptCallResponse

logger = logging.getLogger("humanloop.sdk")


CLIENT_TYPE = TypeVar("CLIENT_TYPE", PromptsClient, FlowsClient)


def overload_log(client: CLIENT_TYPE) -> CLIENT_TYPE:
    """
    Wrap the `log` method of the provided Humanloop client to use EVALUATION_CONTEXT.

    This makes the overloaded log actions be aware of whether the created Log is
    part of an Evaluation (e.g. one started by eval_utils.run_eval).
    """
    # Copy the original log method in a hidden attribute
    client._log = client.log

    def _overload_log(
        self, **kwargs
    ) -> Union[
        CreatePromptLogResponse,
        CreateToolLogResponse,
        CreateFlowLogResponse,
        CreateEvaluatorLogResponse,
    ]:
        if log_belongs_to_evaluated_file(log_args=kwargs):
            evaluation_context = get_evaluation_context()
            for attribute in ["source_datapoint_id", "run_id"]:
                if attribute not in kwargs or kwargs[attribute] is None:
                    kwargs[attribute] = getattr(evaluation_context, attribute)

            # Call the original .log method
            logger.debug(
                "Logging %s inside _overloaded_log on Thread %s",
                kwargs,
                evaluation_context,
                threading.get_ident(),
            )

        try:
            response = self._log(**kwargs)
        except Exception as e:
            logger.error(f"Failed to log: {e}")
            raise e

        # Notify the run_eval utility about one Log being created
        if log_belongs_to_evaluated_file(log_args=kwargs):
            evaluation_context = get_evaluation_context()
            evaluation_context.upload_callback(log_id=response.id)

        return response

    # Replace the original log method with the overloaded one
    client.log = types.MethodType(_overload_log, client)
    # Return the client with the overloaded log method
    logger.debug("Overloaded the .call method of %s", client)
    return client


def overload_prompt_call(client: PromptsClient) -> PromptsClient:
    client._call = client.call

    def _overload_call(self, **kwargs) -> PromptCallResponse:
        if in_prompt_utility_context():
            try:
                response = self._call(**kwargs)
                response = typing.cast(PromptCallResponse, response)
            except Exception as e:
                # TODO: Bug found in backend: not specifying a model 400s but creates a File
                raise HumanloopUtilityError(message=str(e)) from e

            response_copy = response.dict()
            prompt_utility_context = get_prompt_utility_context()
            for idx, _ in enumerate(response_copy.get("logs", [])):
                del response_copy["logs"][idx]["created_at"]
            for idx, _ in enumerate(response_copy["prompt"].get("environments", [])):
                del response_copy["prompt"]["environments"][idx]["created_at"]
            del response_copy["prompt"]["last_used_at"]
            del response_copy["prompt"]["updated_at"]
            del response_copy["prompt"]["created_at"]
            del response_copy["start_time"]
            del response_copy["end_time"]

            with prompt_utility_context.tracer.start_as_current_span(HUMANLOOP_INTERCEPTED_HL_CALL_SPAN_NAME) as span:
                write_to_opentelemetry_span(
                    span=span,
                    key=HUMANLOOP_INTERCEPTED_HL_CALL_RESPONSE,
                    value=response_copy,
                )
            return response
        else:
            return self._call(**kwargs)

    # Replace the original log method with the overloaded one
    client.call = types.MethodType(_overload_call, client)
    # Return the client with the overloaded log method
    logger.debug("Overloaded the .log method of %s", client)
    return client
