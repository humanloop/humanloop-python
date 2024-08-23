# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations
import typing
import typing

if typing.TYPE_CHECKING:
    from .prompt_log_response import PromptLogResponseParams
    from .tool_log_response import ToolLogResponseParams
    from .evaluator_log_response import EvaluatorLogResponseParams
LogResponseParams = typing.Union["PromptLogResponseParams", "ToolLogResponseParams", "EvaluatorLogResponseParams"]