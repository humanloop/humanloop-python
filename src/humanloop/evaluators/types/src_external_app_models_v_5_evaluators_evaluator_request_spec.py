# This file was auto-generated by Fern from our API Definition.

import typing

from ...types.code_evaluator_request import CodeEvaluatorRequest
from ...types.external_evaluator_request import ExternalEvaluatorRequest
from ...types.human_evaluator_request import HumanEvaluatorRequest
from ...types.llm_evaluator_request import LlmEvaluatorRequest

SrcExternalAppModelsV5EvaluatorsEvaluatorRequestSpec = typing.Union[
    LlmEvaluatorRequest, CodeEvaluatorRequest, HumanEvaluatorRequest, ExternalEvaluatorRequest
]
