# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations

import datetime as dt
import typing

import typing_extensions

from .evaluation_result_response_value import EvaluationResultResponseValueParams

if typing.TYPE_CHECKING:
    from .src_external_app_models_v_4_log_log_response import SrcExternalAppModelsV4LogLogResponseParams


class EvaluationResultResponseParams(typing_extensions.TypedDict):
    id: str
    evaluator_id: str
    evaluator_version_id: str
    evaluation_id: typing_extensions.NotRequired[str]
    log_id: str
    log: typing_extensions.NotRequired["SrcExternalAppModelsV4LogLogResponseParams"]
    version_id: typing_extensions.NotRequired[str]
    version: typing_extensions.NotRequired[typing.Any]
    value: typing_extensions.NotRequired[EvaluationResultResponseValueParams]
    error: typing_extensions.NotRequired[str]
    updated_at: dt.datetime
    created_at: dt.datetime
    llm_evaluator_log: typing_extensions.NotRequired["SrcExternalAppModelsV4LogLogResponseParams"]
