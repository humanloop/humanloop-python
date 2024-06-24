# This file was auto-generated by Fern from our API Definition.

import typing

from .code_evaluator_request import CodeEvaluatorRequest
from .human_evaluator_request import HumanEvaluatorRequest
from .llm_evaluator_request import LlmEvaluatorRequest

EvaluatorResponseSpec = typing.Union[LlmEvaluatorRequest, CodeEvaluatorRequest, HumanEvaluatorRequest]
