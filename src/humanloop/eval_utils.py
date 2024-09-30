"""
Evaluation utils for the Humanloop SDK.

This module provides a set of utilities to aid running Eval workflows on Humanloop
where you are managing the runtime of your application in your code.

Functions in this module should be accessed via the Humanloop client. They should
not be called directly.
"""
from datetime import datetime
from functools import partial
import inspect
import json
from logging import Logger, INFO
from pydantic import BaseModel, ValidationError
from typing import Callable, Sequence, Literal
from typing_extensions import NotRequired, TypedDict
import time
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from .client import BaseHumanloop
from .core.api_error import ApiError

# We use TypedDicts for requests, which is consistent with the rest of the SDK
from .requests import FlowKernelRequestParams as FlowDict
from .requests import PromptKernelRequestParams as PromptDict
from .requests import ToolKernelRequestParams as ToolDict
from .requests import CreateDatapointRequestParams as DatapointDict
from .requests import ExternalEvaluatorRequestParams as ExternalEvaluator
from .requests import CodeEvaluatorRequestParams as CodeEvaluatorDict
from .requests import LlmEvaluatorRequestParams as LLMEvaluatorDict
from .requests import HumanEvaluatorRequestParams as HumanEvaluatorDict


# Responses are Pydantic models
from .types import FlowKernelRequest as FlowKernel
from .types import BooleanEvaluatorStatsResponse as BooleanStats
from .types import NumericEvaluatorStatsResponse as NumericStats
from .types import UpdateDatesetAction as UpdateDatasetAction  # TODO: fix original type typo
from .types import (
    EvaluationStats,
    VersionStatsResponse,
    EvaluatorArgumentsType,
    EvaluatorReturnTypeEnum,
    EvaluationResponse
)

# TODO: use logger instead of printing?
logger = Logger(name=__name__, level=INFO)
EvaluatorDict = CodeEvaluatorDict | LLMEvaluatorDict | HumanEvaluatorDict | ExternalEvaluator
Version = FlowDict | PromptDict | ToolDict | EvaluatorDict


# ANSI escape codes for colors
YELLOW = "\033[93m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


class Identifiers(TypedDict, total=False):
    """Common identifiers for the objects required to run an Evaluation."""
    id: NotRequired[str]
    """The ID of the File on Humanloop."""
    path: NotRequired[str]
    """The path of the File on Humanloop."""


class File(Identifiers):
    """A File on Humanloop (Flow, Prompt, Tool, Evaluator)."""
    type: NotRequired[Literal["flow", "prompt", "tool", "evaluator"]]
    """The type of File this pipeline relates to on Humanloop."""
    version: NotRequired[Version]
    """The contents uniquely define the version of the File on Humanloop"""


class RunDataset(Identifiers):
    datapoints: Sequence[DatapointDict]
    """The datapoints to map your pipeline over to produce the outputs required by the evaluation."""
    action: NotRequired[UpdateDatasetAction]
    """How to update the Dataset given the provided Datapoints; 
    `set` replaces the existing Datapoints and `add` appends to the existing Datapoints."""


class RunEvaluator(Identifiers):
    """A to provide judgments for this Evaluation."""
    args_type: NotRequired[EvaluatorArgumentsType]
    """The type of arguments the Evaluator expects - only required for local Evaluators."""
    return_type: NotRequired[EvaluatorReturnTypeEnum]
    """The type of return value the Evaluator produces - only required for local Evaluators."""
    callable: NotRequired[Callable]
    """The function to run on the logs to produce the judgment - only required for local Evaluators."""
    custom_logger: NotRequired[Callable]
    """optional function that logs the output judgment from your Evaluator to Humanloop, if provided, it will be called as follows:
    ```
    judgment = callable(log_dict)
    log = custom_logger(client, judgmemt)
    ```
    Inside the custom_logger, you can use the Humanloop `client` to log the judgment to Humanloop.
    If not provided your pipline must return a single string.
    """
    threshold: NotRequired[float]
    """The threshold to check the Evaluator against."""


class EvaluatorCheck(BaseModel):
    """Summary data for an Evaluator check."""
    path: str
    """The path of the Evaluator used in the check."""
    improvement_check: bool
    """Whether the latest version of your pipeline has improved across for a specific Evaluator."""
    score: float
    """The score of the latest version of your pipeline for a specific Evaluator."""
    delta: float
    """The change in score since the previous version of your pipeline for a specific Evaluator."""
    threshold: float | None
    """The threshold to check the Evaluator against."""
    threshold_check: bool | None
    """Whether the latest version has an average Evaluator result above a threshold."""

# TODO: Explore restarting runs, but without having to regenerate logs


def _run_eval(
    client: BaseHumanloop,
    file: File,
    name: str | None,
    pipeline: Callable,
    dataset: RunDataset,
    evaluators: Sequence[RunEvaluator] | None = None,
    custom_logger: Callable | None = None,
    # logs: typing.Sequence[dict] | None = None,
    workers: int = 5,
) -> list[EvaluatorCheck]:
    """
    Evaluate your `pipeline` for a given `Dataset` and set of `Evaluators`

    :param client: the Humanloop API client.
    :param file: the corresponding Humanloop file where the Evaluation is created and data populated.
    :param pipeline: The function being evaluated.
        It will be called using your Dataset `inputs` as follows: `output = pipeline(**datapoint.inputs)`.
        If `messages` are defined in your Dataset, then `output = pipeline(**datapoint.inputs, messages=datapoint.messages)`.
        It should return a single string output. If not, you must provide a `custom_logger`.
    :param name: the name of the Evaluation to run. If it does not exist, a new Evaluation will be created under your File.
    :param dataset: the dataset to map your pipeline over to produce the outputs required by the Evaluation.
    :param evaluators: define how judgments are provided for this Evaluation.
    :param custom_logger: optional function that logs the output of your pipeline to Humanloop, if provided, it will be called as follows:
        ```
        output = pipeline(**datapoint.inputs).
        log = custom_logger(client, output)
        ```
        Inside the custom_logger, you can use the Humanloop `client` to log the output of your pipeline.
        If not provided your pipline must return a single string.
    :param workers: the number of threads to process datapoints using your pipeline concurrently.
    :return: per Evaluator checks.
    """

    # Get or create the file on Humanloop
    version = file.pop("version", {})
    try:
        type_ = file.pop("type")
    except KeyError as _:
        # Default to flows if not type specified
        type_ = "flow"
        logger.warning("No type specified, defaulting to 'flow'.")

    file_dict = {**file, **version}
    match type_:
        case "flow":
            # Be more lenient with Flow versions as they are arbitrary json
            try:
                FlowKernel.parse_obj(version)
            except ValidationError:
                version = {"attributes": version}
                file_dict = {**file, **version}
            file = client.flows.upsert(**file_dict)
        case "prompt":
            file = client.flows.upsert(**file_dict)
        case "tool":
            file = client.flows.upsert(**file_dict)
        case "evaluator":
            file = client.evaluators.upsert(**file_dict)
        case _:
            raise NotImplementedError(f"Unsupported File type: {type_}")

    # Upsert the Dataset
    hl_dataset = client.datasets.upsert(**dataset)
    hl_dataset = client.datasets.get(id=hl_dataset.id, include_datapoints=True)

    # Upsert the local Evaluators; other Evaluators are just referenced by path
    local_evaluators: list[RunEvaluator] = []
    if evaluators:
        for evaluator in evaluators:
            # If a callable is provided for an Evaluator, we treat it as External
            eval_callable = evaluator.get("callable")
            if eval_callable is not None:
                local_evaluators.append(evaluator)
                spec = ExternalEvaluator(
                    arguments_type=evaluator["args_type"],
                    return_type=evaluator["return_type"],
                    attributes={"code": inspect.getsource(eval_callable)},
                    evaluator_type="external"
                )
                _ = client.evaluators.upsert(
                    id=evaluator["id"],
                    path=evaluator["path"],
                    spec=spec
                )
    # Validate upfront that the local Evaluators and Dataset fit
    requires_target = False
    for local_evaluator in local_evaluators:
        if local_evaluators["args_type"] == "target_required":
            requires_target = True
            break
    if requires_target:
        missing_target = 0
        for datapoint in dataset.datapoints:
            if not datapoint.target:
                missing_target +=1
        if missing_target > 0:
            raise ValueError(f"Datapoint {datapoint.id} has no target is required for this Evaluator: {local_evaluator['path']}")

    # TODO: Should we try a single call to pipeline and local evaluators to check if there are issues before trying to process full dataset?

    # Get or create the Evaluation based on the name
    evaluation = None
    try:
        evaluation = client.evaluations.create(
            name=name,
            dataset={"file_id": hl_dataset.id},
            evaluators=[{"path": e["path"]} for e in evaluators],
            file={"id": file.id},
        )
    except ApiError as error_:
        # If the name exists, go and get it # TODO: Update API GET to allow querying by name and file.
        if error_.status_code == 409:
            evals = client.evaluations.list(file_id=file.id, size=50)
            for page in evals.iter_pages():
                evaluation = next((e for e in page.items if e.name == name), None)
        if not evaluation:
            raise ValueError(f"Evaluation with name {name} not found.")

    # Every run will generate a new batch of logs
    batch_id = uuid.uuid4().hex[:10]  # ignore risk of collision
    log_func = _get_log_func(
        client=client,
        type_=type_,
        file_id=file.id,
        version_id=file.version_id,
        evaluation_id=evaluation.id,
        batch_id=batch_id
    )

    # Define the function to execute your pipeline in parallel and Log to Humanloop
    def process_datapoint(datapoint):
        start_time = datetime.now()
        try:
            if datapoint.messages:
                output = pipeline(**datapoint.inputs, messages=datapoint.messages)
            else:
                output = pipeline(**datapoint.inputs)
            if custom_logger:
                log = custom_logger(client=client, output=output)
            else:
                if not isinstance(output, str):
                    raise ValueError("Your pipeline must return a string if you do not provide a custom logger.")
                log = log_func(
                    inputs=datapoint.inputs,
                    output=output,
                    source_datapoint_id=datapoint.id,
                    start_time=start_time,
                    end_time=datetime.now(),
                )
        except Exception as e:
            log = log_func(
                inputs=datapoint.inputs,
                error=str(e),
                source_datapoint_id=datapoint.id,
                start_time=start_time,
                end_time=datetime.now(),
            )
            logger.warning(msg=f"Pipeline failed for Datapoint: {datapoint.id}. \n Error: {e.args[0]}")

        # Apply local Evaluators
        for local_evaluator in local_evaluators:
            try:
                start_time = datetime.now()
                callable_ = local_evaluator["callable"]
                if local_evaluator["args_type"] == "target_required":
                    judgment = callable_(log.dict(), datapoint.target)
                else:
                    judgment = callable_(log.dict())

                if local_evaluator.get("custom_logger", None):
                    local_evaluator["custom_logger"](client=client, judgment=judgment)
                else:
                    # The API call will validate the judgment
                    _ = client.evaluators.log(
                        parent_id=log.id,
                        id=local_evaluator.get("id"),
                        path=local_evaluator.get("path"),
                        judgment=judgment,
                        start_time=start_time,
                        end_time=datetime.now(),
                    )
            except Exception as e:
                _ = client.evaluators.log(
                    parent_id=log.id,
                    path=local_evaluator.get("path"),
                    id=local_evaluator.get("id"),
                    error=str(e),
                    start_time=start_time,
                    end_time=datetime.now(),
                )
                logger.warning(f"Evaluator {local_evaluator['path']} failed with error {e.args[0]}")

    # Execute the pipeline and send the logs to Humanloop in parallel
    total_datapoints = len(hl_dataset.datapoints)
    print(f"\n{CYAN}Navigate to your Evals:{RESET} {evaluation.url}")
    print(f"{CYAN}\nVersion:{RESET}\n {json.dumps(version, indent=4)}")
    print(f"{CYAN}\nRunning your pipeline over the Dataset...{RESET}")

    completed_tasks = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_datapoint, datapoint)
            for datapoint in hl_dataset.datapoints
        ]
        for _ in as_completed(futures):
            completed_tasks += 1
            _progress_bar(total_datapoints, completed_tasks)

    # Wait for the Evaluation to complete then print the results
    complete = False
    stats = None
    while not complete:
        stats = client.evaluations.get_stats(id=evaluation.id)
        sys.stdout.write(stats.progress)
        sys.stdout.flush()
        complete = stats.status == "completed"
        if not complete:
            time.sleep(5)

    # Print Evaluation results
    print(stats.report)

    checks: list[EvaluatorCheck] = []
    for evaluator in evaluators:
        improvement_check, score, delta = check_evaluation_improvement(
            evaluation=evaluation,
            stats=stats,
            evaluator_path=evaluator["path"],
        )
        threshold_check = None
        threshold = evaluator.get("threshold")
        if threshold is not None:
            threshold_check = check_evaluation_threshold(
                evaluation=evaluation,
                stats=stats,
                evaluator_path=evaluator["path"],
                threshold=threshold,
            )
        checks.append(
            EvaluatorCheck(
                path=evaluator["path"],
                improvement_check=improvement_check,
                score=score,
                delta=delta,
                threshold=threshold,
                threshold_check=threshold_check,
            )
        )
    return checks


def _get_log_func(
        client: BaseHumanloop,
        type_: Literal["flow", "prompt", "tool", "evaluator"],
        file_id: str,
        version_id: str,
        evaluation_id: str,
        batch_id: str,
) -> Callable:
    """Returns the appropriate log function pre-filled with common parameters."""
    log_request = {
        # TODO: why does the Log `id` field refer to the file ID in the API?
        #  Why are both `id` and `version_id` needed in the API?
        "id": file_id,
        "version_id": version_id,
        "evaluation_id": evaluation_id,
        "batch_id": batch_id,
    }
    match type_:
        case "flow":
            return partial(client.flows.log, **log_request, trace_status="complete")
        case "prompt":
            return partial(client.prompts.log, **log_request)
        case "evaluator":
            return partial(client.evaluators.log, **log_request)
        case "tool":
            return partial(client.tools.log, **log_request)
        case _:
            raise NotImplementedError(f"Unsupported File version: {type_}")


def get_score_from_evaluator_stat(stat: NumericStats | BooleanStats) -> float | None:
    """Get the score from an Evaluator Stat."""
    score = None
    match stat:
        case BooleanStats():
            if stat.total_logs:
                score = round(stat.num_true / stat.total_logs, 2)
        case NumericStats():
            score = round(stat.mean, 2)
        case _:
            raise ValueError("Invalid Evaluator Stat type.")
    return score


def _progress_bar(total: int, progress: int):
    """Simple progress bar for CLI with ETA."""

    if total <= 0:
        total = 1

    if not hasattr(_progress_bar, 'start_time'):
        _progress_bar.start_time = time.time()

    bar_length = 40
    block = int(round(bar_length * progress / total))
    bar = "#" * block + '-' * (bar_length - block)

    percentage = (progress / total) * 100
    elapsed_time = time.time() - _progress_bar.start_time
    time_per_item = elapsed_time / progress if progress > 0 else 0
    eta = (total - progress) * time_per_item

    progress_display = f"\r[{bar}] {progress}/{total}"
    progress_display += f" ({percentage:.2f}%)"

    if progress < total:
        progress_display += f" | ETA: {int(eta)}s"
    else:
        progress_display += " | DONE"
        _progress_bar.start_time = None

    sys.stdout.write(progress_display)
    sys.stdout.flush()

    if progress >= total:
        sys.stdout.write("\n")


def get_evaluator_stats_by_path(
    stat: VersionStatsResponse, evaluation: EvaluationResponse
) -> dict[str, NumericStats | BooleanStats]:
    """Get the Evaluator stats by path."""
    # TODO: Update the API so this is not necessary
    evaluators_by_id = {
        evaluator.version.version_id: evaluator for evaluator in evaluation.evaluators
    }
    evaluator_stats_by_path = {
        evaluators_by_id[evaluator_stat.evaluator_version_id].version.path: evaluator_stat
        for evaluator_stat in stat.evaluator_version_stats
    }
    return evaluator_stats_by_path


def check_evaluation_threshold(
    evaluation: EvaluationResponse,
    stats: EvaluationStats,
    evaluator_path: str,
    threshold: float,
) -> bool:
    """Checks if the latest version has an average Evaluator result above a threshold."""
    # TODO: Update the API so this is not necessary
    evaluator_stats_by_path = get_evaluator_stats_by_path(
        stat=stats.version_stats[-1],
        evaluation=evaluation
    )
    if evaluator_path in evaluator_stats_by_path:
        evaluator_stat = evaluator_stats_by_path[evaluator_path]
        score = get_score_from_evaluator_stat(stat=evaluator_stat)
        if score >= threshold:
            print(
                f"{GREEN}✅ Latest eval [{score}] above threshold [{threshold}] for evaluator {evaluator_path}.{RESET}"
            )
            return True
        else:
            print(
                f"{RED}❌ Latest score [{score}] below the threshold [{threshold}] for evaluator {evaluator_path}.{RESET}"
            )
            return False
    else:
        raise ValueError(f"Evaluator {evaluator_path} not found in the stats.")


def check_evaluation_improvement(
    evaluation: EvaluationResponse,
    evaluator_path: str,
    stats: EvaluationStats
) -> tuple[bool, float, float]:
    """
    Check the latest version has improved across for a specific Evaluator.

    :returns: A tuple of (improvement, latest_score, delta since previous score)
    """
    # TODO: Update the API so this is not necessary
    latest_evaluator_stats_by_path = get_evaluator_stats_by_path(
        stat=stats.version_stats[-1],
        evaluation=evaluation
    )
    if len(stats.version_stats) == 1:
        print(
            f"{YELLOW}⚠️ No previous versions to compare with.{RESET}"
        )
        return True, 0, 0

    previous_evaluator_stats_by_path = get_evaluator_stats_by_path(
        stat=stats.version_stats[-2],
        evaluation=evaluation
    )
    if evaluator_path in latest_evaluator_stats_by_path and evaluator_path in previous_evaluator_stats_by_path:
        latest_evaluator_stat = latest_evaluator_stats_by_path[evaluator_path]
        previous_evaluator_stat = previous_evaluator_stats_by_path[evaluator_path]
        latest_score = get_score_from_evaluator_stat(stat=latest_evaluator_stat)
        previous_score = get_score_from_evaluator_stat(stat=previous_evaluator_stat)
        diff = round(latest_score - previous_score, 2)
        if diff >= 0:
            print(
                f"{GREEN}✅ Improvement of [{diff}] for evaluator {evaluator_path}{RESET}"
            )
            return True, latest_score, diff
        else:
            print(
                f"{RED}❌ Regression of [{diff}] for evaluator {evaluator_path}{RESET}"
            )
            return False, latest_score, diff
    else:
        raise ValueError(f"Evaluator {evaluator_path} not found in the stats.")
