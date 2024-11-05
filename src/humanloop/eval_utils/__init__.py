"""
Evaluation utils for the Humanloop SDK.

This module provides a set of utilities to aid running Eval workflows on Humanloop
where you are managing the runtime of your application in your code.

Functions in this module should be accessed via the Humanloop client. They should
not be called directly.
"""

import inspect
import logging
import sys
import threading
import time
import typing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from logging import INFO
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import ValidationError
from typing import Callable, Sequence, Literal, Union, Optional, List, Dict, Tuple
import time
import sys


from humanloop import EvaluatorResponse, FlowResponse, PromptResponse, ToolResponse
from humanloop.client import BaseHumanloop
from humanloop.core.api_error import ApiError
from humanloop.eval_utils.context import EVALUATION_CONTEXT, EvaluationContext
from humanloop.eval_utils.domain import Dataset, Evaluator, EvaluatorCheck, File

# We use TypedDicts for requests, which is consistent with the rest of the SDK
from humanloop.eval_utils.shared import add_log_to_evaluation
from humanloop.requests import CodeEvaluatorRequestParams as CodeEvaluatorDict
from humanloop.requests import ExternalEvaluatorRequestParams as ExternalEvaluator
from humanloop.requests import FlowKernelRequestParams as FlowDict
from humanloop.requests import HumanEvaluatorRequestParams as HumanEvaluatorDict
from humanloop.requests import LlmEvaluatorRequestParams as LLMEvaluatorDict
from humanloop.requests import PromptKernelRequestParams as PromptDict
from humanloop.requests import ToolKernelRequestParams as ToolDict
from humanloop.types import BooleanEvaluatorStatsResponse as BooleanStats
from humanloop.types import DatapointResponse as Datapoint
from humanloop.types import EvaluationResponse, EvaluationStats, VersionStatsResponse

# Responses are Pydantic models and we leverage them for improved request validation
from humanloop.types import FlowKernelRequest as Flow
from humanloop.types import NumericEvaluatorStatsResponse as NumericStats
from humanloop.types import PromptKernelRequest as Prompt
from humanloop.types import ToolKernelRequest as Tool
from humanloop.types import BooleanEvaluatorStatsResponse as BooleanStats
from humanloop.types import NumericEvaluatorStatsResponse as NumericStats
from humanloop.types import DatapointResponse as Datapoint
from humanloop.types import EvaluationStats, EvaluationResponse
from humanloop.types.evaluation_run_response import EvaluationRunResponse
from humanloop.types.run_stats_response import RunStatsResponse

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(level=INFO)
console_handler = logging.StreamHandler()
logger.setLevel(INFO)
formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(console_handler)

EvaluatorDict = Union[CodeEvaluatorDict, LLMEvaluatorDict, HumanEvaluatorDict, ExternalEvaluator]
Version = Union[FlowDict, PromptDict, ToolDict, EvaluatorDict]
FileType = Literal["flow", "prompt", "tool", "evaluator"]


# ANSI escape codes for logging colors
YELLOW = "\033[93m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def _run_eval(
    client: BaseHumanloop,
    file: Union[File, Callable],
    name: Optional[str],
    dataset: Dataset,
    evaluators: Optional[Sequence[Evaluator]] = None,
    # logs: typing.Sequence[dict] | None = None,
    workers: int = 4,
) -> List[EvaluatorCheck]:
    """
    Evaluate your function for a given `Dataset` and set of `Evaluators`.

    :param client: the Humanloop API client.
    :param file: the Humanloop file being evaluated, including a function to run over the dataset.
    :param name: the name of the Evaluation to run. If it does not exist, a new Evaluation will be created under your File.
    :param dataset: the dataset to map your function over to produce the outputs required by the Evaluation.
    :param evaluators: define how judgments are provided for this Evaluation.
    :param workers: the number of threads to process datapoints using your function concurrently.
    :return: per Evaluator checks.
    """
    global _PROGRESS_BAR

    if isinstance(file, Callable):  # type: ignore
        # Decorated function
        file_: File = file.file  # type: ignore
    else:
        file_ = file  # type: ignore

    is_decorated = file_.pop("is_decorated", False)

    # Get or create the file on Humanloop
    version = file_.pop("version", {})

    # Raise error if one of path or id not provided
    if not file_.get("path") and not file_.get("id"):
        raise ValueError("You must provide a path or id in your `file`.")

    # Determine the `type` of the `file` to Evaluate - if not `type` provided, default to `flow`
    try:
        type_ = typing.cast(FileType, file_.pop("type"))
        logger.info(
            f"{CYAN}Evaluating your {type_} function corresponding to `{file_['path']}` on Humanloop{RESET} \n\n"
        )
    except KeyError as _:
        type_ = "flow"
        logger.warning("No `file` type specified, defaulting to flow.")

    # If a `callable` is provided, Logs will be generated locally, otherwise Logs will be generated on Humanloop.
    function_ = typing.cast(Optional[Callable], file_.pop("callable", None))
    if function_ is None:
        if type_ == "flow":
            raise ValueError("You must provide a `callable` for your Flow `file` to run a local eval.")
        else:
            logger.info(f"No `callable` provided for your {type_} file - will attempt to generate logs on Humanloop.")

    custom_logger = file_.pop("custom_logger", None)
    file_dict = {**file_, **version}
    hl_file: Union[PromptResponse, FlowResponse, ToolResponse, EvaluatorResponse]

    if type_ == "flow":
        # Be more lenient with Flow versions as they are arbitrary json
        try:
            Flow.model_validate(version)
        except ValidationError:
            flow_version = {"attributes": version}
            file_dict = {**file_, **flow_version}
        hl_file = client.flows.upsert(**file_dict)  # type: ignore

    elif type_ == "prompt":
        try:
            Prompt.model_validate(version)
        except ValidationError as error_:
            logger.error(msg="Invalid Prompt `version` in your `file` request. \n\nValidation error: \n)")
            raise error_
        hl_file = client.prompts.upsert(**file_dict)  # type: ignore

    elif type_ == "tool":
        try:
            Tool.model_validate(version)
        except ValidationError as error_:
            logger.error(msg="Invalid Tool `version` in your `file` request. \n\nValidation error: \n)")
            raise error_
        hl_file = client.tools.upsert(**file_dict)  # type: ignore

    elif type_ == "evaluator":
        hl_file = client.evaluators.upsert(**file_dict)  # type: ignore

    else:
        raise NotImplementedError(f"Unsupported File type: {type_}")

    # Upsert the Dataset
    action = dataset.get("action", "set")  # set is the server default - None not allowed.
    if "datapoints" not in dataset:
        dataset["datapoints"] = []
        # Use `upsert` to get existing dataset ID if no datapoints provided, given we can't `get` on path.
        action = "add"
    hl_dataset = client.datasets.upsert(**dataset, action=action)
    hl_dataset = client.datasets.get(id=hl_dataset.id, version_id=hl_dataset.version_id, include_datapoints=True)

    # Upsert the local Evaluators; other Evaluators are just referenced by `path` or `id`
    local_evaluators: List[Evaluator] = []
    if evaluators:
        for evaluator in evaluators:
            # If a callable is provided for an Evaluator, we treat it as External
            eval_function = evaluator.get("callable")
            if eval_function is not None:
                # TODO: support the case where `file` logs generated on Humanloop but Evaluator logs generated locally
                if function_ is None:
                    raise ValueError(
                        f"Local Evaluators are only supported when generating Logs locally using your {type_}'s `callable`. Please provide a `callable` for your file in order to run Evaluators locally."
                    )
                local_evaluators.append(evaluator)
                spec = ExternalEvaluator(
                    arguments_type=evaluator["args_type"],
                    return_type=evaluator["return_type"],
                    attributes={"code": inspect.getsource(eval_function)},
                    evaluator_type="external",
                )
                client.evaluators.upsert(
                    id=evaluator.get("id"),
                    path=evaluator.get("path"),
                    spec=spec,
                )
    function_ = typing.cast(Callable, function_)

    # Validate upfront that the local Evaluators and Dataset fit
    requires_target = False
    for local_evaluator in local_evaluators:
        if local_evaluator["args_type"] == "target_required":
            requires_target = True
            break
    if requires_target:
        missing_target = 0
        for datapoint in hl_dataset.datapoints:  # type: ignore
            if not datapoint.target:
                missing_target += 1
        if missing_target > 0:
            raise ValueError(
                f"{missing_target} Datapoints have no target. A target is required for the Evaluator: {local_evaluator['path']}"
            )

    # Get or create the Evaluation based on the name
    evaluation = None
    try:
        evaluation = client.evaluations.create(
            name=name,
            dataset={"file_id": hl_dataset.id},
            evaluators=[{"path": e["path"]} for e in evaluators],  # type: ignore
            file={"id": hl_file.id},
        )
    except ApiError as error_:
        # If the name exists, go and get it # TODO: Update API GET to allow querying by name and file.
        if error_.status_code == 409:
            evals = client.evaluations.list(file_id=hl_file.id, size=50)
            for page in evals.iter_pages():
                evaluation = next((e for e in page.items if e.name == name), None)  # type: ignore
        else:
            raise error_
        if not evaluation:
            raise ValueError(f"Evaluation with name {name} not found.")

    # Create a new Run
    run: EvaluationRunResponse = client.evaluations.create_run(
        id=evaluation.id,
        dataset={"version_id": hl_dataset.version_id},
        orchestrated=False,
    )
    # Every Run will generate a new batch of Logs
    run_id = run.id

    # Define the function to execute your function in parallel and Log to Humanloop
    def process_datapoint(datapoint: Datapoint):
        start_time = datetime.now()
        datapoint_dict = datapoint.dict()
        try:
            if "messages" in datapoint_dict and datapoint_dict["messages"] is not None:
                output = function_(**datapoint_dict["inputs"], messages=datapoint_dict["messages"])
            else:
                function_(datapoint_dict["inputs"])  # type: ignore

    else:
        # Define the function to execute your function in parallel and Log to Humanloop
        def process_datapoint(dp: Datapoint, evaluated_file_id: str, run_id: str):
            log_func = _get_log_func(
                client=client,
                file_type=type_,
                file_id=hl_file.id,
                version_id=hl_file.version_id,
                run_id=run_id,
            )

            start_time = datetime.now()
            datapoint_dict = dp.dict()
            try:
                if "messages" in datapoint_dict:
                    output = function_(  # type: ignore
                        **datapoint_dict["inputs"],
                        messages=datapoint_dict["messages"],
                    )
                else:
                    # function_ is not None at this point
                    output = function_(**datapoint_dict["inputs"])  # type: ignore
                if custom_logger:
                    # function_ is not None at this point
                    log = function_(client=client, output=output)  # type: ignore
                else:
                    if not isinstance(output, str):
                        raise ValueError(
                            f"Your {type_}'s `callable` must return a string if you do not provide a custom logger."
                        )
                    log = log_func(
                        inputs=dp.inputs,
                        output=output,
                        source_datapoint_id=dp.id,
                        start_time=start_time,
                        end_time=datetime.now(),
                    )
            except Exception as e:
                log = log_func(
                    inputs=dp.inputs,
                    error=str(e),
                    source_datapoint_id=dp.id,
                    start_time=start_time,
                    end_time=datetime.now(),
                )
                logger.warning(msg=f"\nYour {type_}'s `callable` failed for Datapoint: {dp.id}. \n Error: {str(e)}")

            add_log_to_evaluation(
                client=client,
                log=log,
                datapoint_target=dp.target,
                local_evaluators=local_evaluators,
            )
            _PROGRESS_BAR.increment()

    # Execute the function and send the logs to Humanloop in parallel
    logger.info(f"\n{CYAN}Navigate to your Evaluation:{RESET}\n{evaluation.url}\n")
    logger.info(f"{CYAN}{type_.capitalize()} Version ID: {hl_file.version_id}{RESET}")
    logger.info(f"{CYAN}Run ID: {run_id}{RESET}")

    # Generate locally if a file `callable` is provided
    if function_:  # type: ignore
        logger.info(
            f"{CYAN}\nRunning '{hl_file.name}' over the Dataset '{hl_dataset.name}' using {workers} workers{RESET} "
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for datapoint in hl_dataset.datapoints:
                executor.submit(
                    process_datapoint,
                    datapoint,
                    hl_file.id,
                    run_id,
                )
    else:
        # TODO: trigger run when updated API is available
        logger.info(f"{CYAN}\nRunning '{hl_file.name}' over the Dataset '{hl_dataset.name}'{RESET}")

    # Wait for the Evaluation to complete then print the results
    complete = False

    while not complete:
        stats = client.evaluations.get_stats(id=evaluation.id)
        logger.info(f"\r{stats.progress}")
        run_stats = next(
            (run_stats for run_stats in stats.run_stats if run_stats.run_id == run_id),
            None,
        )
        complete = run_stats is not None and run_stats.status == "completed"
        if not complete:
            time.sleep(5)

    # Print Evaluation results
    logger.info(stats.report)

    checks: List[EvaluatorCheck] = []

    # Skip `check_evaluation_improvement` if no thresholds were provided and there is only one run.
    # (Or the logs would not be helpful)
    if any(evaluator.get("threshold") is not None for evaluator in evaluators) or len(stats.run_stats) > 1:
        for evaluator in evaluators:
            _, score, delta = check_evaluation_improvement(
                evaluation=evaluation,
                stats=stats,
                evaluator_path=evaluator["path"],
                run_id=run_id,
            )
            threshold_check = None
            threshold = evaluator.get("threshold")
            if threshold is not None:
                threshold_check = check_evaluation_threshold(
                    evaluation=evaluation,
                    stats=stats,
                    evaluator_path=evaluator["path"],
                    threshold=threshold,
                    run_id=run_id,
                )
            checks.append(
                EvaluatorCheck(
                    path=evaluator["path"],
                    # TODO: Add back in with number valence on Evaluators
                    # improvement_check=improvement_check,
                    score=score,
                    delta=delta,
                    threshold=threshold,
                    threshold_check=threshold_check,
                    evaluation_id=evaluation.id,
                )
            )

    logger.info(f"\n{CYAN}View your Evaluation:{RESET}\n{evaluation.url}\n")
    return checks


def _get_log_func(
    client: BaseHumanloop,
    file_type: FileType,
    file_id: str,
    version_id: str,
    run_id: str,
) -> Callable:
    """Returns the appropriate log function pre-filled with common parameters."""
    log_request = {
        # TODO: why does the Log `id` field refer to the file ID in the API?
        #  Why are both `id` and `version_id` needed in the API?
        "id": file_id,
        "version_id": version_id,
        "run_id": run_id,
    }
    if file_type == "flow":
        return partial(client.flows.log, **log_request, trace_status="complete")
    elif file_type == "prompt":
        return partial(client.prompts.log, **log_request)
    elif file_type == "evaluator":
        return partial(client.evaluators.log, **log_request)
    elif file_type == "tool":
        return partial(client.tools.log, **log_request)
    else:
        raise NotImplementedError(f"Unsupported File version: {file_type}")


def get_score_from_evaluator_stat(
    stat: Union[NumericStats, BooleanStats],
) -> Union[float, None]:
    """Get the score from an Evaluator Stat."""
    score = None
    if isinstance(stat, BooleanStats):
        if stat.total_logs:
            score = round(stat.num_true / stat.total_logs, 2)
    elif isinstance(stat, NumericStats):
        score = round(stat.mean, 2)  # type: ignore
    else:
        raise ValueError(f"Unsupported Evaluator Stat type: {type(stat)}")
    return score  # type: ignore


class _SimpleProgressBar:
    def __init__(self, total: int):
        if total <= 0:
            self._total = 1
        else:
            self._total = total
        self._progress = 0
        self._lock = threading.Lock()
        self._start_time = None

    def increment(self):
        with self._lock:
            self._progress += 1
            if self._start_time is None:
                self._start_time = time.time()

            bar_length = 40
            block = int(round(bar_length * self._progress / self._total))
            bar = "#" * block + "-" * (bar_length - block)

            percentage = (self._progress / self._total) * 100
            elapsed_time = time.time() - self._start_time
            time_per_item = elapsed_time / self._progress if self._progress > 0 else 0
            eta = (self._total - self._progress) * time_per_item

            progress_display = f"\r[{bar}] {self._progress}/{self._total}"
            progress_display += f" ({percentage:.2f}%)"

            if self._progress < self._total:
                progress_display += f" | ETA: {int(eta)}s"
            else:
                progress_display += " | DONE"

            sys.stderr.write(progress_display)

            if self._progress >= self._total:
                sys.stderr.write("\n")


_PROGRESS_BAR = None


def get_evaluator_stats_by_path(
    stat: RunStatsResponse,
    evaluation: EvaluationResponse,
) -> Dict[str, Union[NumericStats, BooleanStats]]:
    """Get the Evaluator stats by path."""
    # TODO: Update the API so this is not necessary
    evaluators_by_id = {evaluator.version.version_id: evaluator for evaluator in evaluation.evaluators}
    evaluator_stats_by_path = {
        evaluators_by_id[evaluator_stat.evaluator_version_id].version.path: evaluator_stat
        for evaluator_stat in stat.evaluator_stats
    }
    return evaluator_stats_by_path  # type: ignore


def check_evaluation_threshold(
    evaluation: EvaluationResponse,
    stats: EvaluationStats,
    evaluator_path: str,
    threshold: float,
    run_id: str,
) -> bool:
    """Checks if the latest version has an average Evaluator result above a threshold."""
    # TODO: Update the API so this is not necessary
    evaluator_stats_by_path = get_evaluator_stats_by_path(
        stat=next((stat for stat in stats.run_stats if stat.run_id == run_id), None),
        evaluation=evaluation,
    )
    if evaluator_path in evaluator_stats_by_path:
        evaluator_stat = evaluator_stats_by_path[evaluator_path]
        score = get_score_from_evaluator_stat(stat=evaluator_stat)
        if score >= threshold:
            logger.info(
                f"{GREEN}✅ Latest eval [{score}] above threshold [{threshold}] for evaluator {evaluator_path}.{RESET}"
            )
            return True
        else:
            logger.info(
                f"{RED}❌ Latest score [{score}] below the threshold [{threshold}] for evaluator {evaluator_path}.{RESET}"
            )
            return False
    else:
        raise ValueError(f"Evaluator {evaluator_path} not found in the stats.")


def check_evaluation_improvement(
    evaluation: EvaluationResponse,
    evaluator_path: str,
    stats: EvaluationStats,
    run_id: str,
) -> Tuple[bool, float, float]:
    """
    Check the latest version has improved across for a specific Evaluator.

    :returns: A tuple of (improvement, latest_score, delta since previous score)
    """
    # TODO: Update the API so this is not necessary

    latest_evaluator_stats_by_path = get_evaluator_stats_by_path(
        stat=next((stat for stat in stats.run_stats if stat.run_id == run_id), None),
        evaluation=evaluation,
    )
    if len(stats.run_stats) == 1:
        logger.info(f"{YELLOW}⚠️ No previous versions to compare with.{RESET}")
        return True, 0, 0

    previous_evaluator_stats_by_path = get_evaluator_stats_by_path(
        stat=stats.run_stats[1],  # Latest Run is at index 0; previous Run is at index 1
        evaluation=evaluation,
    )
    if evaluator_path in latest_evaluator_stats_by_path and evaluator_path in previous_evaluator_stats_by_path:
        latest_evaluator_stat = latest_evaluator_stats_by_path[evaluator_path]
        previous_evaluator_stat = previous_evaluator_stats_by_path[evaluator_path]
        latest_score = get_score_from_evaluator_stat(stat=latest_evaluator_stat)
        previous_score = get_score_from_evaluator_stat(stat=previous_evaluator_stat)
        if latest_score is None or previous_score is None:
            raise ValueError(f"Could not find score for Evaluator {evaluator_path}.")
        diff = round(latest_score - previous_score, 2)
        if diff >= 0:
            logger.info(f"{CYAN}Change of [{diff}] for Evaluator {evaluator_path}{RESET}")
            return True, latest_score, diff
        else:
            logger.info(f"{CYAN}Change of [{diff}] for Evaluator {evaluator_path}{RESET}")
            return False, latest_score, diff
    else:
        raise ValueError(f"Evaluator {evaluator_path} not found in the stats.")
