# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations
import typing
import typing

if typing.TYPE_CHECKING:
    from .version_deployment_response import VersionDeploymentResponseParams
    from .version_id_response import VersionIdResponseParams
VersionReferenceResponseParams = typing.Union["VersionDeploymentResponseParams", "VersionIdResponseParams"]
