"""
Evaluation utils for the Humanloop SDK.

This module provides a set of utilities to aid running Eval workflows on Humanloop
where you are managing the runtime of your application in your code.

Functions in this module should be accessed via the Humanloop client. They should
not be called directly.
"""

import copy
from dataclasses import dataclass
import inspect
import json
import logging
import signal
import sys
import threading
import time
import typing
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from functools import partial
from logging import INFO
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from humanloop import EvaluatorResponse, FlowResponse, PromptResponse, ToolResponse
from humanloop.core.api_error import ApiError
from humanloop.context import (
    EvaluationContext,
    get_evaluation_context,
    set_evaluation_context,
)
from humanloop.error import HumanloopRuntimeError
from humanloop.evals.types import Dataset, Evaluator, EvaluatorCheck, File

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
from humanloop.types.datapoint_response import DatapointResponse
from humanloop.types.dataset_response import DatasetResponse
from humanloop.types.evaluation_run_response import EvaluationRunResponse
from humanloop.types.flow_log_response import FlowLogResponse
from humanloop.types.log_response import LogResponse
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


def run_eval(
    client: "BaseHumanloop",
    file: File,
    name: Optional[str],
    dataset: Dataset,
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
    evaluators_worker_pool = ThreadPoolExecutor(max_workers=workers)

    file_ = _file_or_file_inside_hl_utility(file)
    type_ = _get_file_type(file_)
    function_ = _get_file_callable(file_, type_)
    if hasattr(function_, "file"):
        decorator_type = function_.file["type"]  # type: ignore [attr-defined, union-attr]
        if decorator_type != type_:
            raise HumanloopRuntimeError(
                "The type of the decorated function does not match the type of the file. Expected `%s`, got `%s`."
                % (type_.capitalize(), decorator_type.capitalize())
            )

    try:
        hl_file = _upsert_file(file=file_, type=type_, client=client)
    except ValidationError as e:
        sys.stdout.write(f"{RED}Error in your `file` argument:\n\n{e}{RESET}")
        return []
    except Exception as e:
        sys.stdout.write(f"{RED}Error in your `file` argument:\n\n{e}{RESET}")
        return []
    try:
        hl_dataset = _upsert_dataset(dataset=dataset, client=client)
    except Exception as e:
        sys.stdout.write(f"{RED}Error in your `dataset` argument:\n\n{e}{RESET}")
        return []
    try:
        local_evaluators = _upsert_local_evaluators(
            evaluators=evaluators,  # type: ignore [arg-type]
            client=client,
            function=function_,
            type=type_,
        )
    except Exception as e:
        sys.stdout.write(f"{RED}Error in your `evaluators` argument:\n\n{e}{RESET}")
        return []
    _assert_dataset_evaluators_fit(hl_dataset, local_evaluators)

    evaluation, run = _get_new_run(
        client=client,
        evaluation_name=name,
        evaluators=evaluators,  # type: ignore [arg-type]
        hl_file=hl_file,
        hl_dataset=hl_dataset,
        function=function_,
    )

    def _cancel_evaluation():
        client.evaluations.update_evaluation_run(
            id=evaluation.id,
            run_id=run.id,
            status="cancelled",
        )
        evaluators_worker_pool.shutdown(wait=False)

    def handle_exit_signal(signum, frame):
        sys.stderr.write(
            f"\n{RED}Received signal {signum}, cancelling Evaluation and shutting down threads...{RESET}\n"
        )
        _cancel_evaluation()
        sys.exit(signum)

    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    # Header of the CLI Report
    sys.stdout.write(f"\n{CYAN}Navigate to your Evaluation:{RESET}\n{evaluation.url}\n\n")
    sys.stdout.write(f"{CYAN}{type_.capitalize()} Version ID: {hl_file.version_id}{RESET}\n")
    sys.stdout.write(f"{CYAN}Run ID: {run.id}{RESET}\n")

    # This will apply apply the local callable to each datapoint
    # and log the results to Humanloop

    # Generate locally if a file `callable` is provided
    if function_ is None:
        # TODO: trigger run when updated API is available
        sys.stdout.write(f"{CYAN}\nRunning '{hl_file.name}' over the Dataset '{hl_dataset.name}'{RESET}\n")
    else:
        # Running the evaluation locally
        sys.stdout.write(
            f"{CYAN}\nRunning '{hl_file.name}' over the Dataset '{hl_dataset.name}' using {workers} workers...{RESET}\n\n"
        )

    _PROGRESS_BAR = _SimpleProgressBar(len(hl_dataset.datapoints))

    if function_ is not None:
        # Generate locally if a file `callable` is provided
        def _process_datapoint(dp: Datapoint):
            def upload_callback(log_id: str):
                """Logic ran after the Log has been created."""
                # Need to get the full log to pass to the evaluators
                evaluators_worker_pool.submit(
                    _run_local_evaluators,
                    client=client,
                    log_id=log_id,
                    datapoint=dp,
                    local_evaluators=local_evaluators,
                    file_type=hl_file.type,  # type: ignore [arg-type]
                    progress_bar=_PROGRESS_BAR,
                )

            # Set the Evaluation Context for current datapoint
            with set_evaluation_context(
                EvaluationContext(
                    source_datapoint_id=dp.id,
                    eval_callback=upload_callback,
                    file_id=hl_file.id,
                    run_id=run.id,
                    path=hl_file.path,
                )
            ):
                log_func = _get_log_func(
                    client=client,
                    file_type=hl_file.type,  # type: ignore [arg-type]
                    file_id=hl_file.id,
                    version_id=hl_file.version_id,
                    run_id=run.id,
                )
                start_time = datetime.now()
                evaluation_context = get_evaluation_context()
                if evaluation_context is None:
                    raise HumanloopRuntimeError(
                        "Internal error: evaluation context is not set while processing a datapoint."
                    )
                try:
                    output = _call_function(function_, hl_file.type, dp)  # type: ignore [arg-type]
                    if not evaluation_context.logged:
                        # function_ did not Log against the source_datapoint_id/ run_id pair
                        # so we need to create a Log
                        log = log_func(
                            **{
                                "inputs": dp.inputs,
                                "output": output,
                                "start_time": start_time,
                                "end_time": datetime.now(),
                                "source_datapoint_id": dp.id,
                                "run_id": run.id,
                                "log_status": "complete",
                            }
                        )
                        evaluation_context._callback(log.id)
                except HumanloopRuntimeError as e:
                    raise e
                except Exception as e:
                    log = log_func(
                        **{
                            "inputs": dp.inputs,
                            "error": str(e),
                            "source_datapoint_id": dp.id,
                            "run_id": run.id,
                            "start_time": start_time,
                            "end_time": datetime.now(),
                            "log_status": "complete",
                        }
                    )
                    evaluation_context._callback(log.id)
                    error_message = _get_error_message(e, length_limit=True)
                    sys.stderr.write(
                        f"\n{RED}Evaluated callable failed for Datapoint `{dp.id}`:\n{error_message}{RESET}\n"
                    )

        futures: list[Future] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for datapoint in hl_dataset.datapoints:
                futures.append(executor.submit(_process_datapoint, datapoint))

        for future in futures:
            try:
                future.result()
            except Exception as e:
                sys.stderr.write(f"\n{RED}Error processing datapoint:\n{_get_error_message(e)}{RESET}\n")
                _cancel_evaluation()
                return []

    stats = _wait_for_evaluation_to_complete(
        client=client,
        evaluation=evaluation,
        run=run,
    )
    sys.stderr.write(f"\n{CYAN}View your Evaluation:{RESET}\n{evaluation.url}\n")

    # Print Evaluation results
    sys.stderr.write(stats.report)

    checks = _get_checks(
        client=client,
        evaluation=evaluation,
        stats=stats,
        evaluators=evaluators,  # type: ignore [arg-type]
        run=run,
    )
    evaluators_worker_pool.shutdown(wait=False)
    return checks


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
            if self._progress == self._total:
                return
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


@dataclass
class _LocalEvaluator:
    hl_evaluator: EvaluatorResponse
    function: Callable


def _callable_is_hl_utility(file: File) -> bool:
    """Check if a File is a decorated function."""
    return hasattr(file.get("callable", {}), "file")


def _wait_for_evaluation_to_complete(
    client: "BaseHumanloop",
    evaluation: EvaluationResponse,
    run: EvaluationRunResponse,
):
    # Wait for the Evaluation to complete then print the results
    complete = False

    waiting_for_local_evals_message_printed = False

    while not complete:
        stats = client.evaluations.get_stats(id=evaluation.id)
        run_stats = next(
            (run_stats for run_stats in stats.run_stats if run_stats.run_id == run.id),
            None,
        )
        complete = run_stats is not None and run_stats.status == "completed"
        if not complete:
            if not waiting_for_local_evals_message_printed:
                sys.stderr.write("\n\nWaiting for Evaluators on Humanloop runtime...\n")
                waiting_for_local_evals_message_printed = True
            sys.stderr.write(stats.progress)  # type: ignore [arg-type]
            # Move the cursor up in stderr a number of lines equal to the number of lines in stats.progress
            # sys.stderr.write("\033[A" * (stats.progress.count("\n")))
            time.sleep(5)

    return stats


def _get_checks(
    client: "BaseHumanloop",
    evaluation: EvaluationResponse,
    stats: EvaluationStats,
    evaluators: list[Evaluator],
    run: EvaluationRunResponse,
):
    checks: List[EvaluatorCheck] = []

    # Skip `check_evaluation_improvement` if no thresholds were provided and there is only one run.
    # (Or the logs would not be helpful)
    if any(evaluator.get("threshold") is not None for evaluator in evaluators) or len(stats.run_stats) > 1:
        for evaluator in evaluators:
            score, delta = _check_evaluation_improvement(
                evaluation=evaluation,
                stats=stats,
                evaluator_path=evaluator["path"],
                run_id=run.id,
            )[1:]
            threshold_check = None
            threshold = evaluator.get("threshold")
            if threshold is not None:
                threshold_check = _check_evaluation_threshold(
                    evaluation=evaluation,
                    stats=stats,
                    evaluator_path=evaluator["path"],
                    threshold=threshold,
                    run_id=run.id,
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

    return checks


def _file_or_file_inside_hl_utility(file: File) -> File:
    if _callable_is_hl_utility(file):
        inner_file: File = file["callable"].file  # type: ignore [misc, attr-defined]
        if "path" in file and inner_file["path"] != file["path"]:
            raise HumanloopRuntimeError(
                "`path` attribute specified in the `file` does not match the path of the decorated function. "
                f"Expected `{inner_file['path']}`, got `{file['path']}`."
            )
        if "id" in file:
            raise HumanloopRuntimeError(
                "Do not specify an `id` attribute in `file` argument when using a decorated function."
            )
        if "version" in file:
            if inner_file["type"] != "prompt":
                raise HumanloopRuntimeError(
                    f"Do not specify a `version` attribute in `file` argument when using a {inner_file['type'].capitalize()} decorated function."
                )
        if "type" in file and inner_file["type"] != file["type"]:
            raise HumanloopRuntimeError(
                "Attribute `type` of `file` argument does not match the file type of the decorated function. "
                f"Expected `{inner_file['type']}`, got `{file['type']}`."
            )
        if "id" in file:
            raise HumanloopRuntimeError(
                "Do not specify an `id` attribute in `file` argument when using a decorated function."
            )
        # file on decorated function holds at least
        # or more information than the `file` argument
        file_ = copy.deepcopy(inner_file)
        if file_["type"] == "prompt":
            sys.stdout.write(
                f"{YELLOW}"
                "The @prompt decorator will not spy on provider calls when passed to `evaluations.run()`. "
                "Using the `version` in `file` argument instead.\n"
                f"{RESET}"
            )
            # TODO: document this
            file_["version"] = file["version"]
    else:
        file_ = copy.deepcopy(file)

    # Raise error if neither path nor id is provided
    if not file_.get("path") and not file_.get("id"):
        raise HumanloopRuntimeError("You must provide a path or id in your `file`.")

    return file_


def _get_file_type(file: File) -> FileType:
    # Determine the `type` of the `file` to Evaluate - if not `type` provided, default to `flow`
    try:
        type_ = typing.cast(FileType, file.pop("type"))  # type: ignore [arg-type, misc]
        sys.stdout.write(
            f"{CYAN}Evaluating your {type_} function corresponding to `{file.get('path') or file.get('id')}` on Humanloop{RESET}\n\n"
        )
        return type_ or "flow"
    except KeyError as _:
        type_ = "flow"
        sys.stdout.write(f"{YELLOW}No `file` type specified, defaulting to flow.{RESET}\n")
        return type_


def _get_file_callable(file: File, type_: FileType) -> Optional[Callable]:
    # Get the `callable` from the `file` to Evaluate
    function_ = typing.cast(Optional[Callable], file.pop("callable", None))
    if function_ is None:
        if type_ == "flow":
            raise ValueError("You must provide a `callable` for your Flow `file` to run a local eval.")
        else:
            sys.stdout.write(
                f"{CYAN}No `callable` provided for your {type_} file - will attempt to generate logs on Humanloop.\n\n{RESET}"
            )
    return function_


def _upsert_file(
    file: File, type: FileType, client: "BaseHumanloop"
) -> Union[PromptResponse, FlowResponse, ToolResponse, EvaluatorResponse]:
    # Get or create the file on Humanloop
    version = file.pop("version", {})
    file_dict = {**file, **version}

    if type == "flow":
        # Be more lenient with Flow versions as they are arbitrary json
        try:
            Flow.model_validate(version)
        except ValidationError:
            flow_version = {"attributes": version}
            file_dict = {**file, **flow_version}
        hl_file = client.flows.upsert(**file_dict)  # type: ignore [arg-type, assignment]

    elif type == "prompt":
        # Will throw error if version is invalid
        Prompt.model_validate(version)
        hl_file = client.prompts.upsert(**file_dict)  # type: ignore [arg-type, assignment]

    elif type == "tool":
        # Will throw error if version is invalid
        Tool.model_validate(version)
        hl_file = client.tools.upsert(**file_dict)  # type: ignore [arg-type, assignment]

    elif type == "evaluator":
        hl_file = client.evaluators.upsert(**file_dict)  # type: ignore [arg-type, assignment]

    else:
        raise NotImplementedError(f"Unsupported File type: {type}")

    return hl_file


def _upsert_dataset(dataset: Dataset, client: "BaseHumanloop"):
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
    return client.datasets.get(
        id=hl_dataset.id,
        version_id=hl_dataset.version_id,
        include_datapoints=True,
    )


def _upsert_local_evaluators(
    evaluators: list[Evaluator],
    function: Optional[Callable],
    type: FileType,
    client: "BaseHumanloop",
) -> list[_LocalEvaluator]:
    # Upsert the local Evaluators; other Evaluators are just referenced by `path` or `id`
    local_evaluators: list[_LocalEvaluator] = []
    if evaluators:
        for evaluator_request in evaluators:
            # If a callable is provided for an Evaluator, we treat it as External
            eval_function = evaluator_request.get("callable")
            if eval_function is not None:
                # TODO: support the case where `file` logs generated on Humanloop but Evaluator logs generated locally
                if function is None:
                    raise ValueError(
                        "Local Evaluators are only supported when generating Logs locally using your "
                        f"{type}'s `callable`. Please provide a `callable` for your file in order "
                        "to run Evaluators locally."
                    )
                spec = ExternalEvaluator(
                    arguments_type=evaluator_request["args_type"],
                    return_type=evaluator_request["return_type"],
                    attributes={"code": inspect.getsource(eval_function)},
                    evaluator_type="external",
                )
                try:
                    evaluator = client.evaluators.upsert(
                        id=evaluator_request.get("id"),
                        path=evaluator_request.get("path"),
                        spec=spec,
                    )
                except Exception as error_:
                    sys.stdout.write(
                        f"Error upserting Evaluator {evaluator_request.get('path') or evaluator_request.get('id')} on Humanloop:\n\n{error_}"
                    )
                    raise error_
                local_evaluators.append(_LocalEvaluator(hl_evaluator=evaluator, function=eval_function))
    return local_evaluators


def _assert_dataset_evaluators_fit(
    hl_dataset: DatasetResponse,
    local_evaluators: list[_LocalEvaluator],
):
    # Validate upfront that the local Evaluators and Dataset fit
    requires_target = False
    for hl_evaluator in [local_evaluator.hl_evaluator for local_evaluator in local_evaluators]:
        if hl_evaluator.spec.arguments_type == "target_required":
            requires_target = True
            break
    if requires_target:
        missing_target = 0
        for _datapoint in hl_dataset.datapoints:  # type: ignore [union-attr]
            if not _datapoint.target:
                missing_target += 1
        if missing_target > 0:
            raise ValueError(
                f"{missing_target} Datapoints have no target. A target "
                f"is required for the Evaluator: {hl_evaluator.path}"
            )


def _get_new_run(
    client: "BaseHumanloop",
    evaluation_name: Optional[str],
    evaluators: list[Evaluator],
    hl_file: Union[PromptResponse, FlowResponse, ToolResponse, EvaluatorResponse],
    hl_dataset: DatasetResponse,
    function: Optional[Callable],
):
    # Get or create the Evaluation based on the name
    evaluation = None
    try:
        evaluation = client.evaluations.create(
            name=evaluation_name,
            evaluators=[{"path": e["path"]} for e in evaluators],
            file={"id": hl_file.id},
        )
    except ApiError as error_:
        # If the name exists, go and get it # TODO: Update API GET to allow querying by name and file.
        if error_.status_code == 409:
            evals = client.evaluations.list(file_id=hl_file.id, size=50)
            for page in evals.iter_pages():
                evaluation = next((e for e in page.items if e.name == evaluation_name), None)  # type: ignore [union-attr]
        else:
            raise error_
        if not evaluation:
            raise ValueError(f"Evaluation with name {evaluation_name} not found.")
    # Create a new Run
    run: EvaluationRunResponse = client.evaluations.create_run(
        id=evaluation.id,
        dataset={"version_id": hl_dataset.version_id},
        version={"version_id": hl_file.version_id},
        orchestrated=False if function is not None else True,
        use_existing_logs=False,
    )
    return evaluation, run


def _call_function(
    function: Callable,
    type: FileType,
    datapoint: DatapointResponse,
) -> str:
    datapoint_dict = datapoint.dict()
    if "messages" in datapoint_dict and datapoint_dict["messages"] is not None:
        output = function(
            **datapoint_dict["inputs"],
            messages=datapoint_dict["messages"],
        )
    else:
        output = function(**datapoint_dict["inputs"])

    if not isinstance(output, str):
        try:
            output = json.dumps(output)
        except Exception:
            # throw error if it fails to serialize
            raise ValueError(f"Your {type}'s `callable` must return a string or a JSON serializable object.")
    return output


def _get_log_func(
    client: "BaseHumanloop",
    file_type: Literal["flow", "prompt"],
    file_id: str,
    version_id: str,
    run_id: str,
) -> Callable[..., LogResponse]:
    """Returns the appropriate log function pre-filled with common parameters."""
    log_request = {
        # TODO: why does the Log `id` field refer to the file ID in the API?
        #  Why are both `id` and `version_id` needed in the API?
        "id": file_id,
        "version_id": version_id,
        "run_id": run_id,
    }
    if file_type == "flow":
        return partial(client.flows._log, **log_request)  # type: ignore [attr-defined]
    elif file_type == "prompt":
        return partial(client.prompts._log, **log_request)  # type: ignore [attr-defined]
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
        score = round(stat.mean, 2)  # type: ignore [arg-type]
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
    return evaluator_stats_by_path  # type: ignore [return-value]


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
            None,  # type: ignore [arg-type]
        ),
        evaluation=evaluation,
    )
    if evaluator_path in evaluator_stats_by_path:
        evaluator_stat = evaluator_stats_by_path[evaluator_path]
        score = _get_score_from_evaluator_stat(stat=evaluator_stat)
        if score >= threshold:  # type: ignore [operator]
            sys.stderr.write(
                f"{GREEN}✅ Latest eval [{score}] above threshold [{threshold}] for evaluator {evaluator_path}.{RESET}"
            )
            return True
        else:
            sys.stderr.write(
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
            None,  # type: ignore [arg-type]
        ),
        evaluation=evaluation,
    )
    if len(stats.run_stats) == 1:
        sys.stderr.write(f"{YELLOW}⚠️ No previous versions to compare with.{RESET}\n")
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
            sys.stderr.write(f"{CYAN}Change of [{diff}] for Evaluator {evaluator_path}{RESET}\n")
            return True, latest_score, diff
        else:
            sys.stderr.write(f"{CYAN}Change of [{diff}] for Evaluator {evaluator_path}{RESET}\n")
            return False, latest_score, diff
    else:
        raise ValueError(f"Evaluator {evaluator_path} not found in the stats.")


def _get_error_message(e: Exception, length_limit: bool = False) -> str:
    import traceback

    # Get the full traceback
    trace_info = traceback.format_exc()

    # Extract the last 200 characters of the traceback
    last_trace_part = (
        (trace_info[-200:] + "..." if len(trace_info) > 200 else trace_info) if length_limit else trace_info
    )

    return f"\n{last_trace_part}"


def _run_local_evaluators(
    client: "BaseHumanloop",
    log_id: str,
    datapoint: Optional[Datapoint],
    local_evaluators: list[_LocalEvaluator],
    file_type: FileType,
    progress_bar: _SimpleProgressBar,
):
    """Run local Evaluators on the Log and send the judgments to Humanloop."""
    try:
        # Need to get the full log to pass to the evaluators
        log = client.logs.get(id=log_id)
        if not isinstance(log, dict):
            log_dict = log.dict()
        else:
            log_dict = log

        # Wait for the Flow trace to complete before running evaluators
        while True:
            if file_type != "flow" or log_dict["log_status"] == "complete":
                break
            log = client.logs.get(id=log_id)
            if not isinstance(log, dict):
                log_dict = log.dict()
            else:
                log_dict = log
            time.sleep(2)
        datapoint_dict = datapoint.dict() if datapoint else None

        for local_evaluator_tuple in local_evaluators:
            eval_function = local_evaluator_tuple.function
            local_evaluator = local_evaluator_tuple.hl_evaluator
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
                    start_time=start_time,
                    end_time=datetime.now(),
                )
            except Exception as e:
                _ = client.evaluators.log(
                    parent_id=log_id,
                    id=local_evaluator.id,
                    error=_get_error_message(e, length_limit=True),
                    start_time=start_time,
                    end_time=datetime.now(),
                )
                error_message = _get_error_message(e, length_limit=True)
                sys.stderr.write(f"{RED}Evaluator `{local_evaluator.path}` failed: {error_message}{RESET}\n")
    except Exception as e:
        error_message = _get_error_message(e, length_limit=True)
        sys.stderr.write(
            f"{RED}Failed to run local Evaluators for source datapoint `{datapoint.dict()['id'] if datapoint else None}`:\n{error_message}{RESET}\n"
        )
        pass
    finally:
        progress_bar.increment()
