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
import pydantic
from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.pydantic_utilities import update_forward_refs


class ListPrompts(UncheckedBaseModel):
    records: typing.List[PromptResponse] = pydantic.Field()
    """
    The list of Prompts.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow


update_forward_refs(AgentLinkedFileResponse, ListPrompts=ListPrompts)
update_forward_refs(AgentResponse, ListPrompts=ListPrompts)
update_forward_refs(EvaluatorResponse, ListPrompts=ListPrompts)
update_forward_refs(FlowResponse, ListPrompts=ListPrompts)
update_forward_refs(MonitoringEvaluatorResponse, ListPrompts=ListPrompts)
update_forward_refs(PromptResponse, ListPrompts=ListPrompts)
update_forward_refs(ToolResponse, ListPrompts=ListPrompts)
update_forward_refs(VersionDeploymentResponse, ListPrompts=ListPrompts)
update_forward_refs(VersionIdResponse, ListPrompts=ListPrompts)
