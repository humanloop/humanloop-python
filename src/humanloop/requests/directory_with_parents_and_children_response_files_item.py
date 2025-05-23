# This file was auto-generated by Fern from our API Definition.

import typing

from .agent_response import AgentResponseParams
from .dataset_response import DatasetResponseParams
from .evaluator_response import EvaluatorResponseParams
from .flow_response import FlowResponseParams
from .prompt_response import PromptResponseParams
from .tool_response import ToolResponseParams

DirectoryWithParentsAndChildrenResponseFilesItemParams = typing.Union[
    PromptResponseParams,
    ToolResponseParams,
    EvaluatorResponseParams,
    DatasetResponseParams,
    FlowResponseParams,
    AgentResponseParams,
]
