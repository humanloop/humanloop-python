# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ..core.datetime_utils import serialize_datetime
from ..core.pydantic_utilities import deep_union_pydantic_dicts, pydantic_v1
from ..core.unchecked_base_model import UncheckedBaseModel
from .evaluatee_request import EvaluateeRequest
from .evaluations_dataset_request import EvaluationsDatasetRequest
from .evaluations_request import EvaluationsRequest


class CreateEvaluationRequest(UncheckedBaseModel):
    """
    Request model for creating an Evaluation.

    Evaluation benchmark your Prompt/Tool Versions. With the Datapoints in a Dataset Version,
    Logs corresponding to the Datapoint and each Evaluated Version are evaluated by the specified Evaluator Versions.
    Aggregated statistics are then calculated and presented in the Evaluation.
    """

    dataset: EvaluationsDatasetRequest = pydantic_v1.Field()
    """
    The Dataset Version to use in this Evaluation.
    """

    evaluatees: typing.List[EvaluateeRequest] = pydantic_v1.Field()
    """
    Unique identifiers for the Prompt/Tool Versions to include in the Evaluation Report.
    """

    evaluators: typing.List[EvaluationsRequest] = pydantic_v1.Field()
    """
    The Evaluators used to evaluate.
    """

    def json(self, **kwargs: typing.Any) -> str:
        kwargs_with_defaults: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        return super().json(**kwargs_with_defaults)

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        kwargs_with_defaults_exclude_unset: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        kwargs_with_defaults_exclude_none: typing.Any = {"by_alias": True, "exclude_none": True, **kwargs}

        return deep_union_pydantic_dicts(
            super().dict(**kwargs_with_defaults_exclude_unset), super().dict(**kwargs_with_defaults_exclude_none)
        )

    class Config:
        frozen = True
        smart_union = True
        extra = pydantic_v1.Extra.allow
        json_encoders = {dt.datetime: serialize_datetime}
