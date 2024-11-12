from typing import Callable, Literal, Optional, Sequence, TypedDict, Union

from pydantic import BaseModel
from typing_extensions import NotRequired

from humanloop.requests import CodeEvaluatorRequestParams as CodeEvaluatorDict
from humanloop.requests import CreateDatapointRequestParams as DatapointDict
from humanloop.requests import ExternalEvaluatorRequestParams as ExternalEvaluator

# We use TypedDicts for requests, which is consistent with the rest of the SDK
from humanloop.requests import FlowKernelRequestParams as FlowDict
from humanloop.requests import HumanEvaluatorRequestParams as HumanEvaluatorDict
from humanloop.requests import LlmEvaluatorRequestParams as LLMEvaluatorDict
from humanloop.requests import PromptKernelRequestParams as PromptDict
from humanloop.requests import ToolKernelRequestParams as ToolDict
from humanloop.types import (
    EvaluatorArgumentsType,
    EvaluatorReturnTypeEnum,
)

# Responses are Pydantic models and we leverage them for improved request validation
from humanloop.types import UpdateDatesetAction as UpdateDatasetAction  # TODO: fix original type typo

EvaluatorDict = Union[CodeEvaluatorDict, LLMEvaluatorDict, HumanEvaluatorDict, ExternalEvaluator]
Version = Union[FlowDict, PromptDict, ToolDict, EvaluatorDict]
FileType = Literal["flow", "prompt", "tool", "evaluator"]


class Identifiers(TypedDict):
    """Common identifiers for the objects required to run an Evaluation."""

    id: NotRequired[str]
    """The ID of the File on Humanloop."""
    path: NotRequired[str]
    """The path of the File on Humanloop."""


class File(Identifiers):
    """A File on Humanloop (Flow, Prompt, Tool, Evaluator)."""

    type: NotRequired[FileType]
    """The type of File this callable relates to on Humanloop."""
    version: NotRequired[Version]
    """The contents uniquely define the version of the File on Humanloop."""
    callable: NotRequired[Callable]
    """The function being evaluated.
    It will be called using your Dataset `inputs` as follows: `output = callable(**datapoint.inputs)`.
    If `messages` are defined in your Dataset, then `output = callable(**datapoint.inputs, messages=datapoint.messages)`.
    It should return a string or json serializable output.
    """


class Dataset(Identifiers):
    datapoints: NotRequired[Sequence[DatapointDict]]
    """The datapoints to map your function over to produce the outputs required by the evaluation."""
    action: NotRequired[UpdateDatasetAction]
    """How to update the Dataset given the provided Datapoints; 
    `set` replaces the existing Datapoints and `add` appends to the existing Datapoints."""


class Evaluator(Identifiers):
    """The Evaluator to provide judgments for this Evaluation."""

    args_type: NotRequired[EvaluatorArgumentsType]
    """The type of arguments the Evaluator expects - only required for local Evaluators."""
    return_type: NotRequired[EvaluatorReturnTypeEnum]
    """The type of return value the Evaluator produces - only required for local Evaluators."""
    callable: NotRequired[Callable]
    """The function to run on the logs to produce the judgment - only required for local Evaluators."""
    threshold: NotRequired[float]
    """The threshold to check the Evaluator against. If the aggregate value of the Evaluator is below this threshold, the check will fail."""


class EvaluatorCheck(BaseModel):
    """Summary data for an Evaluator check."""

    path: str
    """The path of the Evaluator used in the check."""
    # TODO: Add number valence and improvement check
    # improvement_check: bool
    # """Whether the latest version of your function has improved across the Dataset for a specific Evaluator."""
    score: float
    """The score of the latest version of your function for a specific Evaluator."""
    delta: float
    """The change in score since the previous version of your function for a specific Evaluator."""
    threshold: Optional[float]
    """The threshold to check the Evaluator against."""
    threshold_check: Optional[bool]
    """Whether the latest version has an average Evaluator result above a threshold."""
    evaluation_id: str
    """The ID of the corresponding Evaluation."""
