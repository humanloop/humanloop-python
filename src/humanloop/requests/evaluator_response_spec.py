# This file was auto-generated by Fern from our API Definition.

import typing

from .code_evaluator_request import CodeEvaluatorRequestParams
from .external_evaluator_request import ExternalEvaluatorRequestParams
from .human_evaluator_request import HumanEvaluatorRequestParams
from .llm_evaluator_request import LlmEvaluatorRequestParams

EvaluatorResponseSpecParams = typing.Union[
    LlmEvaluatorRequestParams, CodeEvaluatorRequestParams, HumanEvaluatorRequestParams, ExternalEvaluatorRequestParams
]
