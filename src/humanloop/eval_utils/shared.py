import logging
import typing
from datetime import datetime

from humanloop.base_client import BaseHumanloop
from humanloop.eval_utils.domain import Evaluator
from humanloop.types.datapoint_response_target_value import DatapointResponseTargetValue

logger = logging.getLogger("humanloop.sdk")


def add_log_to_evaluation(
    client: BaseHumanloop,
    log: dict,
    datapoint_target: typing.Optional[typing.Dict[str, DatapointResponseTargetValue]],
    local_evaluators: list[Evaluator],
):
    for local_evaluator in local_evaluators:
        start_time = datetime.now()
        try:
            eval_function = local_evaluator["callable"]
            if local_evaluator["args_type"] == "target_required":
                judgement = eval_function(
                    log,
                    datapoint_target,
                )
            else:
                judgement = eval_function(log)

            if local_evaluator.get("custom_logger", None):
                local_evaluator["custom_logger"](judgement, start_time, datetime.now())
            else:
                _ = client.evaluators.log(
                    parent_id=log['id'],
                    judgment=judgement,
                    id=local_evaluator.get("id"),
                    path=local_evaluator.get("path"),
                    start_time=start_time,
                    end_time=datetime.now(),
                )
        except Exception as e:
            _ = client.evaluators.log(
                parent_id=log['id'],
                path=local_evaluator.get("path"),
                id=local_evaluator.get("id"),
                error=str(e),
                start_time=start_time,
                end_time=datetime.now(),
            )
            logger.warning(f"\nEvaluator {local_evaluator['path']} failed with error {str(e)}")
