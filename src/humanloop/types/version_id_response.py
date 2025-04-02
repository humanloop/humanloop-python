# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations
from ..core.unchecked_base_model import UncheckedBaseModel
import pydantic
import typing
from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.pydantic_utilities import update_forward_refs


class VersionIdResponse(UncheckedBaseModel):
    """
    A reference to a specific Version by its ID
    """

    version: "VersionIdResponseVersion" = pydantic.Field()
    """
    The specific Version being referenced.
    """

    type: typing.Literal["version"] = "version"

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow


from .agent_linked_file_response import AgentLinkedFileResponse  # noqa: E402
from .agent_response import AgentResponse  # noqa: E402
from .evaluator_response import EvaluatorResponse  # noqa: E402
from .flow_response import FlowResponse  # noqa: E402
from .monitoring_evaluator_response import MonitoringEvaluatorResponse  # noqa: E402
from .prompt_response import PromptResponse  # noqa: E402
from .tool_response import ToolResponse  # noqa: E402
from .version_deployment_response import VersionDeploymentResponse  # noqa: E402
from .version_id_response_version import VersionIdResponseVersion  # noqa: E402

update_forward_refs(AgentLinkedFileResponse, VersionIdResponse=VersionIdResponse)
update_forward_refs(AgentResponse, VersionIdResponse=VersionIdResponse)
update_forward_refs(EvaluatorResponse, VersionIdResponse=VersionIdResponse)
update_forward_refs(FlowResponse, VersionIdResponse=VersionIdResponse)
update_forward_refs(MonitoringEvaluatorResponse, VersionIdResponse=VersionIdResponse)
update_forward_refs(PromptResponse, VersionIdResponse=VersionIdResponse)
update_forward_refs(ToolResponse, VersionIdResponse=VersionIdResponse)
update_forward_refs(VersionDeploymentResponse, VersionIdResponse=VersionIdResponse)
update_forward_refs(VersionIdResponse)
