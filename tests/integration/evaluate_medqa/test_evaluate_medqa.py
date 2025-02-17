import time
from typing import Callable

import pytest
from humanloop.types.evaluation_response import EvaluationResponse
from tests.conftest import DirectoryIdentifiers
from tests.integration.evaluate_medqa.conftest import MedQAScenario
from humanloop import Humanloop


@pytest.mark.skip("skip for demo")
@pytest.mark.parametrize("use_call", [True])
def test_scenario(
    evaluate_medqa_scenario_factory: Callable[[bool], MedQAScenario],
    humanloop_client: Humanloop,
    test_directory: DirectoryIdentifiers,
    use_call: bool,
):
    evaluate_medqa_scenario = evaluate_medqa_scenario_factory(use_call)
    ask_question_path, ask_question = evaluate_medqa_scenario.ask_question
    medqa_dataset_path, medqa_dataset = evaluate_medqa_scenario.medqa_dataset_path
    levenshtein_path = evaluate_medqa_scenario.levenshtein_path
    exact_match_path = evaluate_medqa_scenario.exact_match_path

    humanloop_client.evaluations.run(  # type: ignore [attr-defined]
        name="Test",
        file={
            "path": ask_question_path,
            "callable": ask_question,
        },
        dataset={
            "path": medqa_dataset_path,
            "datapoints": medqa_dataset[:1],
        },
        evaluators=[
            {"path": levenshtein_path},
            {"path": exact_match_path},
        ],
    )

    time.sleep(3)

    response = humanloop_client.directories.get(test_directory.id)
    flow = [file for file in response.files if file.type == "flow"][0]
    logs_page = humanloop_client.logs.list(file_id=flow.id)
    assert len(logs_page.items) == 1  # type: ignore [arg-type]

    flow_log_id = logs_page.items[0].id  # type: ignore [index]
    flow_log = humanloop_client.logs.get(flow_log_id)
    if not isinstance(flow_log, dict):
        flow_log = flow_log.dict()  # type: ignore [assignment]
    assert flow_log["trace_status"] == "complete"  # type: ignore [index]
    assert len(flow_log["trace_children"]) == 2  # type: ignore [index]

    levenshtein = [file for file in response.files if file.path == levenshtein_path][0]
    levenshtein_logs_page = humanloop_client.logs.list(file_id=levenshtein.id)
    assert len(levenshtein_logs_page.items) == 1  # type: ignore [arg-type]
    assert levenshtein_logs_page.items[0].parent_id == flow_log_id  # type: ignore
    assert levenshtein_logs_page.items[0].error is None  # type: ignore [index]

    exact_match = [file for file in response.files if file.path == exact_match_path][0]
    exact_match_logs_page = humanloop_client.logs.list(file_id=exact_match.id)
    assert len(exact_match_logs_page.items) == 1  # type: ignore [arg-type]
    assert exact_match_logs_page.items[0].parent_id == flow_log_id  # type: ignore
    assert exact_match_logs_page.items[0].error is None  # type: ignore [index]

    response = humanloop_client.evaluations.list(file_id=flow.id)  # type: ignore [assignment]
    assert len(response.items) == 1  # type: ignore [attr-defined]
    evaluation: EvaluationResponse = response.items[0]  # type: ignore [attr-defined]
    assert evaluation.status == "completed"  # type: ignore [attr-defined]
    assert evaluation.name == "Test"
    assert evaluation.runs_count == 1
    assert evaluation.file_id == flow.id
    for evaluator in evaluation.evaluators:
        assert evaluator.orchestrated is True
