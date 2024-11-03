# This file was auto-generated by Fern from our API Definition.

import typing
from ...requests.llm_evaluator_request import LlmEvaluatorRequestParams
from ...requests.code_evaluator_request import CodeEvaluatorRequestParams
from ...requests.human_evaluator_request import HumanEvaluatorRequestParams
from ...requests.external_evaluator_request import ExternalEvaluatorRequestParams

EvaluatorRequestSpecParams = typing.Union[
    LlmEvaluatorRequestParams, CodeEvaluatorRequestParams, HumanEvaluatorRequestParams, ExternalEvaluatorRequestParams
]