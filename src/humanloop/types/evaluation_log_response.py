# This file was auto-generated by Fern from our API Definition.

from ..core.unchecked_base_model import UncheckedBaseModel
from .evaluator_log_response import EvaluatorLogResponse
from .evaluator_response import EvaluatorResponse
from .flow_log_response import FlowLogResponse
from .flow_response import FlowResponse
from .monitoring_evaluator_response import MonitoringEvaluatorResponse
from .prompt_log_response import PromptLogResponse
from .prompt_response import PromptResponse
from .tool_log_response import ToolLogResponse
from .tool_response import ToolResponse
from .version_deployment_response import VersionDeploymentResponse
from .version_id_response import VersionIdResponse
import pydantic
import typing
from .datapoint_response import DatapointResponse
from .log_response import LogResponse
from ..core.pydantic_utilities import IS_PYDANTIC_V2


class EvaluationLogResponse(UncheckedBaseModel):
    run_id: str = pydantic.Field()
    """
    Unique identifier for the Run.
    """

    datapoint: typing.Optional[DatapointResponse] = pydantic.Field(default=None)
    """
    The Datapoint used to generate the Log
    """

    log: LogResponse = pydantic.Field()
    """
    The Log that was evaluated by the Evaluator.
    """

    evaluator_logs: typing.List[LogResponse] = pydantic.Field()
    """
    The Evaluator Logs containing the judgments for the Log.
    """

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
