# This file was auto-generated by Fern from our API Definition.

import typing

from .version_deployment_response import VersionDeploymentResponse
from .version_id_response import VersionIdResponse

VersionReferenceResponse = typing.Union[VersionDeploymentResponse, VersionIdResponse]
