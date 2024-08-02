# This file was auto-generated by Fern from our API Definition.

import typing

from ...requests.code_evaluator_request import CodeEvaluatorRequestParams
from ...requests.external_evaluator_request import ExternalEvaluatorRequestParams
from ...requests.human_evaluator_request import HumanEvaluatorRequestParams
from ...requests.llm_evaluator_request import LlmEvaluatorRequestParams

CreateEvaluatorLogRequestSpecParams = typing.Union[
    LlmEvaluatorRequestParams, CodeEvaluatorRequestParams, HumanEvaluatorRequestParams, ExternalEvaluatorRequestParams
]