# This file was auto-generated by Fern from our API Definition.

import typing

from .agent_config_response import AgentConfigResponse
from .evaluator_config_response import EvaluatorConfigResponse
from .generic_config_response import GenericConfigResponse
from .model_config_response import ModelConfigResponse
from .tool_config_response import ToolConfigResponse

ConfigResponse = typing.Union[
    ModelConfigResponse, ToolConfigResponse, EvaluatorConfigResponse, AgentConfigResponse, GenericConfigResponse
]
