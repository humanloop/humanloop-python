# This file was auto-generated by Fern from our API Definition.

import typing
from .prompt_response import PromptResponse
from .tool_response import ToolResponse
from .evaluator_response import EvaluatorResponse
from .dataset_response import DatasetResponse
from .flow_response import FlowResponse

DirectoryWithParentsAndChildrenResponseFilesItem = typing.Union[
    PromptResponse, ToolResponse, EvaluatorResponse, DatasetResponse, FlowResponse
]
