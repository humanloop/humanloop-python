# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations
from ..core.unchecked_base_model import UncheckedBaseModel
from .agent_linked_file_response import AgentLinkedFileResponse
from .agent_response import AgentResponse
from .evaluator_response import EvaluatorResponse
from .flow_response import FlowResponse
from .monitoring_evaluator_response import MonitoringEvaluatorResponse
from .prompt_response import PromptResponse
from .tool_response import ToolResponse
from .version_deployment_response import VersionDeploymentResponse
from .version_id_response import VersionIdResponse
import typing
from ..core.pydantic_utilities import IS_PYDANTIC_V2
import pydantic
from ..core.pydantic_utilities import update_forward_refs


class PaginatedDataPromptResponse(UncheckedBaseModel):
    records: typing.List[PromptResponse]
    page: int
    size: int
    total: int

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow


update_forward_refs(AgentLinkedFileResponse, PaginatedDataPromptResponse=PaginatedDataPromptResponse)
update_forward_refs(AgentResponse, PaginatedDataPromptResponse=PaginatedDataPromptResponse)
update_forward_refs(EvaluatorResponse, PaginatedDataPromptResponse=PaginatedDataPromptResponse)
update_forward_refs(FlowResponse, PaginatedDataPromptResponse=PaginatedDataPromptResponse)
update_forward_refs(MonitoringEvaluatorResponse, PaginatedDataPromptResponse=PaginatedDataPromptResponse)
update_forward_refs(PromptResponse, PaginatedDataPromptResponse=PaginatedDataPromptResponse)
update_forward_refs(ToolResponse, PaginatedDataPromptResponse=PaginatedDataPromptResponse)
update_forward_refs(VersionDeploymentResponse, PaginatedDataPromptResponse=PaginatedDataPromptResponse)
update_forward_refs(VersionIdResponse, PaginatedDataPromptResponse=PaginatedDataPromptResponse)
