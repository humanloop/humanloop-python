"""
Evaluation utils for the Humanloop SDK.

This module provides a set of utilities to aid running Eval workflows on Humanloop
where you are managing the runtime of your application in your code.

Functions in this module should be accessed via the Humanloop client. They should
not be called directly.
"""

import copy
import inspect
import json
import logging
import sys
import threading
import time
import types
import typing
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from datetime import datetime
from functools import partial
from logging import INFO
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypeVar, Union

from humanloop import EvaluatorResponse, FlowResponse, PromptResponse, ToolResponse
from humanloop.core.api_error import ApiError
from humanloop.eval_utils.context import EvaluationContext
from humanloop.eval_utils.types import Dataset, Evaluator, EvaluatorCheck, File

# We use TypedDicts for requests, which is consistent with the rest of the SDK
from humanloop.evaluators.client import EvaluatorsClient
from humanloop.flows.client import FlowsClient
from humanloop.prompts.client import PromptsClient
from humanloop.requests import CodeEvaluatorRequestParams as CodeEvaluatorDict
from humanloop.requests import ExternalEvaluatorRequestParams as ExternalEvaluator
from humanloop.requests import FlowKernelRequestParams as FlowDict
from humanloop.requests import HumanEvaluatorRequestParams as HumanEvaluatorDict
from humanloop.requests import LlmEvaluatorRequestParams as LLMEvaluatorDict
from humanloop.requests import PromptKernelRequestParams as PromptDict
from humanloop.requests import ToolKernelRequestParams as ToolDict
from humanloop.tools.client import ToolsClient
from humanloop.types import BooleanEvaluatorStatsResponse as BooleanStats
from humanloop.types import DatapointResponse as Datapoint
from humanloop.types import EvaluationResponse, EvaluationStats

# Responses are Pydantic models and we leverage them for improved request validation
from humanloop.types import FlowKernelRequest as Flow
from humanloop.types import NumericEvaluatorStatsResponse as NumericStats
from humanloop.types import PromptKernelRequest as Prompt
from humanloop.types import ToolKernelRequest as Tool
from humanloop.types.create_evaluator_log_response import CreateEvaluatorLogResponse
from humanloop.types.create_flow_log_response import CreateFlowLogResponse
from humanloop.types.create_prompt_log_response import CreatePromptLogResponse
from humanloop.types.create_tool_log_response import CreateToolLogResponse
from humanloop.types.evaluation_run_response import EvaluationRunResponse
from humanloop.types.run_stats_response import RunStatsResponse
from pydantic import ValidationError

if typing.TYPE_CHECKING:
    from humanloop.client import BaseHumanloop

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


CLIENT_TYPE = TypeVar("CLIENT_TYPE", PromptsClient, ToolsClient, FlowsClient, EvaluatorsClient)


def log_with_evaluation_context(
    client: CLIENT_TYPE,
    evaluation_context_variable: ContextVar[Optional[EvaluationContext]],
) -> CLIENT_TYPE:
    """
    Wrap the `log` method of the provided Humanloop client to use EVALUATION_CONTEXT.

    This makes the overloaded log actions be aware of whether the created Log is
    part of an Evaluation (e.g. one started by eval_utils.run_eval).
    """

    def _is_evaluated_file(
        evaluation_context: EvaluationContext,
        log_args: dict,
    ) -> bool:
        """Check if the File that will Log against is part of the current Evaluation.

        The user of the .log API can refer to the File that owns that Log either by
        ID or Path. This function matches against any of them in EvaluationContext.
        """
        if evaluation_context is None:
            return False
        return evaluation_context.get("file_id") == log_args.get("id") or evaluation_context.get(
            "path"
        ) == log_args.get("path")

    # Copy the original log method in a hidden attribute
    client._log = client.log

    def _overload_log(
        self,
        **kwargs,
    ) -> Union[
        CreatePromptLogResponse,
        CreateToolLogResponse,
        CreateFlowLogResponse,
        CreateEvaluatorLogResponse,
    ]:
        try:
            evaluation_context = evaluation_context_variable.get()
        except LookupError:
            # If the Evaluation Context is not set, an Evaluation is not running
            evaluation_context = None

        if _is_evaluated_file(evaluation_context=evaluation_context, log_args=kwargs):
            # If the .log API user does not provide the source_datapoint_id or run_id,
            # override them with the values from the EvaluationContext
            # _is_evaluated_file ensures that evaluation_context is not None
            for attribute in ["source_datapoint_id", "run_id"]:
                if attribute not in kwargs or kwargs[attribute] is None:
                    kwargs[attribute] = evaluation_context[attribute]

        # Call the original .log method
        logger.debug(
            "Logging %s inside _overloaded_log on Thread %s",
            kwargs,
            evaluation_context,
            threading.get_ident(),
        )
        response = self._log(**kwargs)

        if _is_evaluated_file(
            evaluation_context=evaluation_context,
            log_args=kwargs,
        ):
            # Call the callback so the Evaluation can be updated
            # _is_evaluated_file ensures that evaluation_context is not None
            evaluation_context["upload_callback"](log_id=response.id)

            # Mark the Evaluation Context as consumed
            evaluation_context_variable.set(None)

        return response

    # Replace the original log method with the overloaded one
    client.log = types.MethodType(_overload_log, client)
    # Return the client with the overloaded log method
    logger.debug("Overloaded the .log method of %s", client)
    return client


class _SimpleProgressBar:
    """Thread-safe progress bar for the console."""

    def __init__(self, total: int):
        if total <= 0:
            self._total = 1
        else:
            self._total = total
        self._progress = 0
        self._lock = threading.Lock()
        self._start_time = None

    def increment(self):
        """Increment the progress bar by one finished task."""
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

            progress_display = f"[{bar}] {self._progress}/{self._total}"
            progress_display += f" ({percentage:.2f}%)"

            if self._progress < self._total:
                progress_display += f" | ETA: {int(eta)}s"
            else:
                progress_display += " | DONE"

            sys.stderr.write("\r")  # Move the cursor to the beginning of the line
            sys.stderr.write("\033[K")  # Clear the line from the cursor to the end
            sys.stderr.write(progress_display)

            if self._progress >= self._total:
                sys.stderr.write("\n")


# Module-level so it can be shared by threads.
_PROGRESS_BAR: Optional[_SimpleProgressBar] = None


def run_eval(
    client: "BaseHumanloop",
    file: File,
    name: Optional[str],
    dataset: Dataset,
    evaluation_context_variable: ContextVar[Optional[EvaluationContext]],
    evaluators: Optional[Sequence[Evaluator]] = None,
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

    if hasattr(file["callable"], "file"):
        # When the decorator inside `file` is a decorated function,
        # we need to validate that the other parameters of `file`
        # match the attributes of the decorator
        inner_file: File = file["callable"].file
        if "path" in file and inner_file["path"] != file["path"]:
            raise ValueError(
                "`path` attribute specified in the `file` does not match the File path of the decorated function."
            )
        if "version" in file and inner_file["version"] != file["version"]:
            raise ValueError(
                "`version` attribute in the `file` does not match the File version of the decorated function."
            )
        if "type" in file and inner_file["type"] != file["type"]:
            raise ValueError(
                "`type` attribute of `file` argument does not match the File type of the decorated function."
            )
        if "id" in file:
            raise ValueError("Do not specify an `id` attribute in `file` argument when using a decorated function.")
        # file on decorated function holds at least
        # or more information than the `file` argument
        file_ = copy.deepcopy(inner_file)
    else:
        file_ = file

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

    file_dict = {**file_, **version}
    hl_file: Union[PromptResponse, FlowResponse, ToolResponse, EvaluatorResponse]

    if type_ == "flow":
        # Be more lenient with Flow versions as they are arbitrary json
        try:
            Flow.model_validate(version)
        except ValidationError:
            flow_version = {"attributes": version}
            file_dict = {**file_, **flow_version}
        hl_file = client.flows.upsert(**file_dict)

    elif type_ == "prompt":
        try:
            Prompt.model_validate(version)
        except ValidationError as error_:
            logger.error(msg="Invalid Prompt `version` in your `file` request. \n\nValidation error: \n)")
            raise error_
        try:
            hl_file = client.prompts.upsert(**file_dict)
        except ApiError as error_:
            raise error_

    elif type_ == "tool":
        try:
            Tool.model_validate(version)
        except ValidationError as error_:
            logger.error(msg="Invalid Tool `version` in your `file` request. \n\nValidation error: \n)")
            raise error_
        hl_file = client.tools.upsert(**file_dict)

    elif type_ == "evaluator":
        hl_file = client.evaluators.upsert(**file_dict)

    else:
        raise NotImplementedError(f"Unsupported File type: {type_}")

    # Upsert the Dataset
    if "action" not in dataset:
        dataset["action"] = "set"
    if "datapoints" not in dataset:
        dataset["datapoints"] = []
        # Use `upsert` to get existing dataset ID if no datapoints provided, given we can't `get` on path.
        dataset["action"] = "add"
    hl_dataset = client.datasets.upsert(
        **dataset,
    )
    hl_dataset = client.datasets.get(
        id=hl_dataset.id,
        version_id=hl_dataset.version_id,
        include_datapoints=True,
    )

    # Upsert the local Evaluators; other Evaluators are just referenced by `path` or `id`
    local_evaluators: List[tuple[EvaluatorResponse, Callable]] = []
    if evaluators:
        for evaluator_request in evaluators:
            # If a callable is provided for an Evaluator, we treat it as External
            eval_function = evaluator_request.get("callable")
            if eval_function is not None:
                # TODO: support the case where `file` logs generated on Humanloop but Evaluator logs generated locally
                if function_ is None:
                    raise ValueError(
                        "Local Evaluators are only supported when generating Logs locally using your "
                        f"{type_}'s `callable`. Please provide a `callable` for your file in order "
                        "to run Evaluators locally."
                    )
                spec = ExternalEvaluator(
                    arguments_type=evaluator_request["args_type"],
                    return_type=evaluator_request["return_type"],
                    attributes={"code": inspect.getsource(eval_function)},
                    evaluator_type="external",
                )
                evaluator = client.evaluators.upsert(
                    id=evaluator_request.get("id"),
                    path=evaluator_request.get("path"),
                    spec=spec,
                )
                local_evaluators.append((evaluator, eval_function))

    # function_ cannot be None, cast it for type checking
    function_ = typing.cast(Callable, function_)

    # Validate upfront that the local Evaluators and Dataset fit
    requires_target = False
    for local_evaluator, _ in local_evaluators:
        if local_evaluator.spec.arguments_type == "target_required":
            requires_target = True
            break
    if requires_target:
        missing_target = 0
        for _datapoint in hl_dataset.datapoints:
            if not _datapoint.target:
                missing_target += 1
        if missing_target > 0:
            raise ValueError(
                f"{missing_target} Datapoints have no target. A target "
                f"is required for the Evaluator: {local_evaluator.path}"
            )

    # Get or create the Evaluation based on the name
    evaluation = None
    try:
        evaluation = client.evaluations.create(
            name=name,
            evaluators=[{"path": e["path"]} for e in evaluators],
            file={"id": hl_file.id},
        )
    except ApiError as error_:
        # If the name exists, go and get it # TODO: Update API GET to allow querying by name and file.
        if error_.status_code == 409:
            evals = client.evaluations.list(file_id=hl_file.id, size=50)
            for page in evals.iter_pages():
                evaluation = next((e for e in page.items if e.name == name), None)
        else:
            raise error_
        if not evaluation:
            raise ValueError(f"Evaluation with name {name} not found.")

    # Create a new Run
    run: EvaluationRunResponse = client.evaluations.create_run(
        id=evaluation.id,
        dataset={"version_id": hl_dataset.version_id},
        version={"version_id": hl_file.version_id},
        orchestrated=False if function_ is not None else True,
        use_existing_logs=False,
    )
    # Every Run will generate a new batch of Logs
    run_id = run.id

    _PROGRESS_BAR = _SimpleProgressBar(len(hl_dataset.datapoints))

    # Define the function to execute the `callable` in parallel and Log to Humanloop
    def process_datapoint(dp: Datapoint, file_id: str, file_path: str, run_id: str):
        def upload_callback(log_id: str):
            """Logic ran after the Log has been created."""
            _run_local_evaluators(
                client=client,
                log_id=log_id,
                datapoint=dp,
                local_evaluators=local_evaluators,
            )
            _PROGRESS_BAR.increment()

        datapoint_dict = dp.dict()
        # Set the Evaluation Context for current datapoint
        evaluation_context_variable.set(
            EvaluationContext(
                source_datapoint_id=dp.id,
                upload_callback=upload_callback,
                file_id=file_id,
                run_id=run_id,
                path=file_path,
            )
        )
        logger.debug(
            "process_datapoint on Thread %s: evaluating Datapoint %s with EvaluationContext %s",
            threading.get_ident(),
            datapoint_dict,
            # .get() is safe since process_datapoint is always called in the context of an Evaluation
            evaluation_context_variable.get(),
        )
        # TODO: shouldn't this only be defined in case where we actually need to log?
        log_func = _get_log_func(
            client=client,
            file_type=type_,
            file_id=hl_file.id,
            version_id=hl_file.version_id,
            run_id=run_id,
        )
        start_time = datetime.now()
        try:
            if "messages" in datapoint_dict and datapoint_dict["messages"] is not None:
                output = function_(
                    **datapoint_dict["inputs"],
                    messages=datapoint_dict["messages"],
                )
            else:
                output = function_(**datapoint_dict["inputs"])

            if not isinstance(output, str):
                try:
                    output = json.dumps(output)
                except Exception:
                    # throw error if it fails to serialize
                    raise ValueError(f"Your {type_}'s `callable` must return a string or a JSON serializable object.")

            # .get() is safe since process_datapoint is always called in the context of an Evaluation
            context_variable = evaluation_context_variable.get()
            if context_variable is not None:
                # Evaluation Context has not been consumed
                # function_ is a plain callable so we need to create a Log
                logger.debug(
                    "process_datapoint on Thread %s: function_ %s is a simple callable, context was not consumed",
                    threading.get_ident(),
                    function_.__name__,
                )
                log_func(
                    inputs=dp.inputs,
                    output=output,
                    start_time=start_time,
                    end_time=datetime.now(),
                )
        except Exception as e:
            log_func(
                inputs=dp.inputs,
                error=str(e),
                source_datapoint_id=dp.id,
                run_id=run_id,
                start_time=start_time,
                end_time=datetime.now(),
            )
            logger.warning(msg=f"\nYour {type_}'s `callable` failed for Datapoint: {dp.id}. \n Error: {str(e)}")

    # Execute the function and send the logs to Humanloop in parallel
    logger.info(f"\n{CYAN}Navigate to your Evaluation:{RESET}\n{evaluation.url}\n")
    logger.info(f"{CYAN}{type_.capitalize()} Version ID: {hl_file.version_id}{RESET}")
    logger.info(f"{CYAN}Run ID: {run_id}{RESET}")

    # Generate locally if a file `callable` is provided
    if function_:
        logger.info(
            f"{CYAN}\nRunning '{hl_file.name}' over the Dataset '{hl_dataset.name}' using {workers} workers{RESET} "
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for datapoint in hl_dataset.datapoints:
                executor.submit(
                    process_datapoint,
                    datapoint,
                    hl_file.id,
                    hl_file.path,
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
            score, delta = _check_evaluation_improvement(
                evaluation=evaluation,
                stats=stats,
                evaluator_path=evaluator["path"],
                run_id=run_id,
            )[1:]
            threshold_check = None
            threshold = evaluator.get("threshold")
            if threshold is not None:
                threshold_check = _check_evaluation_threshold(
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
    client: "BaseHumanloop",
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


def _get_score_from_evaluator_stat(
    stat: Union[NumericStats, BooleanStats],
) -> Union[float, None]:
    """Get the score from an Evaluator Stat."""
    score = None
    if isinstance(stat, BooleanStats):
        if stat.total_logs:
            score = round(stat.num_true / stat.total_logs, 2)
    elif isinstance(stat, NumericStats):
        score = round(stat.mean, 2)
    else:
        raise ValueError(f"Unsupported Evaluator Stat type: {type(stat)}")
    return score


def _get_evaluator_stats_by_path(
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
    return evaluator_stats_by_path


def _check_evaluation_threshold(
    evaluation: EvaluationResponse,
    stats: EvaluationStats,
    evaluator_path: str,
    threshold: float,
    run_id: str,
) -> bool:
    """Checks if the latest version has an average Evaluator result above a threshold."""
    # TODO: Update the API so this is not necessary
    evaluator_stats_by_path = _get_evaluator_stats_by_path(
        stat=next(
            (stat for stat in stats.run_stats if stat.run_id == run_id),
            None,
        ),
        evaluation=evaluation,
    )
    if evaluator_path in evaluator_stats_by_path:
        evaluator_stat = evaluator_stats_by_path[evaluator_path]
        score = _get_score_from_evaluator_stat(stat=evaluator_stat)
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


def _check_evaluation_improvement(
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

    latest_evaluator_stats_by_path = _get_evaluator_stats_by_path(
        stat=next(
            (stat for stat in stats.run_stats if stat.run_id == run_id),
            None,
        ),
        evaluation=evaluation,
    )
    if len(stats.run_stats) == 1:
        logger.info(f"{YELLOW}⚠️ No previous versions to compare with.{RESET}")
        return True, 0, 0

    previous_evaluator_stats_by_path = _get_evaluator_stats_by_path(
        stat=stats.run_stats[1],  # Latest Run is at index 0; previous Run is at index 1
        evaluation=evaluation,
    )
    if evaluator_path in latest_evaluator_stats_by_path and evaluator_path in previous_evaluator_stats_by_path:
        latest_evaluator_stat = latest_evaluator_stats_by_path[evaluator_path]
        previous_evaluator_stat = previous_evaluator_stats_by_path[evaluator_path]
        latest_score = _get_score_from_evaluator_stat(stat=latest_evaluator_stat)
        previous_score = _get_score_from_evaluator_stat(stat=previous_evaluator_stat)
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


def _run_local_evaluators(
    client: "BaseHumanloop",
    log_id: str,
    datapoint: Optional[Datapoint],
    local_evaluators: list[tuple[EvaluatorResponse, Callable]],
):
    """Run local Evaluators on the Log and send the judgments to Humanloop."""
    # Need to get the full log to pass to the evaluators
    log = client.logs.get(id=log_id)
    if not isinstance(log, dict):
        log_dict = log.dict()
    else:
        log_dict = log
    datapoint_dict = datapoint.dict() if datapoint else None
    for local_evaluator, eval_function in local_evaluators:
        start_time = datetime.now()
        try:
            if local_evaluator.spec.arguments_type == "target_required":
                judgement = eval_function(
                    log_dict,
                    datapoint_dict,
                )
            else:
                judgement = eval_function(log_dict)

            _ = client.evaluators.log(
                version_id=local_evaluator.version_id,
                parent_id=log_id,
                judgment=judgement,
                id=local_evaluator.id,
                path=local_evaluator.path,
                start_time=start_time,
                end_time=datetime.now(),
            )
        except Exception as e:
            _ = client.evaluators.log(
                parent_id=log_id,
                path=local_evaluator.path,
                id=local_evaluator.id,
                error=str(e),
                start_time=start_time,
                end_time=datetime.now(),
            )
            logger.warning(f"\nEvaluator {local_evaluator.path} failed with error {str(e)}")
