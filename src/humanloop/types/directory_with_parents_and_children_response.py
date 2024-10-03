# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations
from ..core.unchecked_base_model import UncheckedBaseModel
from .evaluator_response import EvaluatorResponse
from .flow_response import FlowResponse
from .monitoring_evaluator_response import MonitoringEvaluatorResponse
from .prompt_response import PromptResponse
from .tool_response import ToolResponse
from .version_deployment_response import VersionDeploymentResponse
from .version_id_response import VersionIdResponse
import pydantic
import typing
import datetime as dt
from .directory_response import DirectoryResponse
from .directory_with_parents_and_children_response_files_item import DirectoryWithParentsAndChildrenResponseFilesItem
from ..core.pydantic_utilities import IS_PYDANTIC_V2
from ..core.pydantic_utilities import update_forward_refs


class DirectoryWithParentsAndChildrenResponse(UncheckedBaseModel):
    id: str = pydantic.Field()
    """
    String ID of directory. Starts with `dir_`.
    """

    parent_id: typing.Optional[str] = pydantic.Field(default=None)
    """
    ID of the parent directory. Will be `None` if the directory is the root directory. Starts with `dir_`.
    """

    name: str = pydantic.Field()
    """
    Name of the directory.
    """

    path: str = pydantic.Field()
    """
    Path to the directory, relative to the root directory. Includes name, e.g. `path/to/directory`.
    """

    created_at: dt.datetime
    updated_at: dt.datetime
    subdirectories: typing.List[DirectoryResponse] = pydantic.Field()
    """
    List of subdirectories in the directory.
    """

    files: typing.List[DirectoryWithParentsAndChildrenResponseFilesItem] = pydantic.Field()
    """
    List of files in the directory.
    """

    parents: typing.List[DirectoryResponse] = pydantic.Field()
    """
    List of parent directories of the directory.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow


update_forward_refs(EvaluatorResponse, DirectoryWithParentsAndChildrenResponse=DirectoryWithParentsAndChildrenResponse)
update_forward_refs(FlowResponse, DirectoryWithParentsAndChildrenResponse=DirectoryWithParentsAndChildrenResponse)
update_forward_refs(
    MonitoringEvaluatorResponse, DirectoryWithParentsAndChildrenResponse=DirectoryWithParentsAndChildrenResponse
)
update_forward_refs(PromptResponse, DirectoryWithParentsAndChildrenResponse=DirectoryWithParentsAndChildrenResponse)
update_forward_refs(ToolResponse, DirectoryWithParentsAndChildrenResponse=DirectoryWithParentsAndChildrenResponse)
update_forward_refs(
    VersionDeploymentResponse, DirectoryWithParentsAndChildrenResponse=DirectoryWithParentsAndChildrenResponse
)
update_forward_refs(VersionIdResponse, DirectoryWithParentsAndChildrenResponse=DirectoryWithParentsAndChildrenResponse)
