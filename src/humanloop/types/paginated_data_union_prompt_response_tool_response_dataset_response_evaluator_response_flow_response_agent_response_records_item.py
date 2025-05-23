# This file was auto-generated by Fern from our API Definition.

import typing

from .agent_response import AgentResponse
from .dataset_response import DatasetResponse
from .evaluator_response import EvaluatorResponse
from .flow_response import FlowResponse
from .prompt_response import PromptResponse
from .tool_response import ToolResponse

PaginatedDataUnionPromptResponseToolResponseDatasetResponseEvaluatorResponseFlowResponseAgentResponseRecordsItem = (
    typing.Union[PromptResponse, ToolResponse, DatasetResponse, EvaluatorResponse, FlowResponse, AgentResponse]
)
