# This file was auto-generated by Fern from our API Definition.

import typing
from ...requests.prompt_response import PromptResponseParams
from ...requests.tool_response import ToolResponseParams
from ...requests.dataset_response import DatasetResponseParams
from ...requests.evaluator_response import EvaluatorResponseParams
from ...requests.flow_response import FlowResponseParams

RetrieveByPathFilesRetrieveByPathPostResponseParams = typing.Union[
    PromptResponseParams, ToolResponseParams, DatasetResponseParams, EvaluatorResponseParams, FlowResponseParams
]
