# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations

import typing

import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2, update_forward_refs
from ..core.unchecked_base_model import UncheckedBaseModel
from .linked_file_request import LinkedFileRequest
from .on_agent_call_enum import OnAgentCallEnum


class AgentLinkedFileResponse(UncheckedBaseModel):
    type: typing.Literal["file"] = "file"
    link: LinkedFileRequest
    on_agent_call: typing.Optional[OnAgentCallEnum] = None
    file: typing.Optional["AgentLinkedFileResponseFile"] = None

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow


from .agent_response import AgentResponse  # noqa: E402, F401, I001
from .evaluator_response import EvaluatorResponse  # noqa: E402, F401, I001
from .flow_response import FlowResponse  # noqa: E402, F401, I001
from .monitoring_evaluator_response import MonitoringEvaluatorResponse  # noqa: E402, F401, I001
from .prompt_response import PromptResponse  # noqa: E402, F401, I001
from .tool_response import ToolResponse  # noqa: E402, F401, I001
from .version_deployment_response import VersionDeploymentResponse  # noqa: E402, F401, I001
from .version_id_response import VersionIdResponse  # noqa: E402, F401, I001
from .agent_linked_file_response_file import AgentLinkedFileResponseFile  # noqa: E402, F401, I001

update_forward_refs(AgentLinkedFileResponse)
