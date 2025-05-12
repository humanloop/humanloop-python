import time
from typing import Any

import pytest
from humanloop.error import HumanloopRuntimeError
from tests.custom.integration.conftest import ResourceIdentifiers
from tests.custom.types import GetHumanloopClientFn


def test_eval_run_works_on_online_files(
    get_humanloop_client: GetHumanloopClientFn,
    output_not_null_evaluator: ResourceIdentifiers,
    eval_dataset: ResourceIdentifiers,
    eval_prompt: ResourceIdentifiers,
) -> None:
    humanloop_client = get_humanloop_client()
    humanloop_client.evaluations.run(  # type: ignore [attr-defined]
        name="test_eval_run",
        file={
            "path": eval_prompt.file_path,
            "type": "prompt",
        },
        dataset={
            "path": eval_dataset.file_path,
        },
        evaluators=[
            {
                "path": output_not_null_evaluator.file_path,
            }
        ],
    )
    time.sleep(5)
    response = humanloop_client.evaluations.list(file_id=eval_prompt.file_id)
    assert response.items and len(response.items) == 1
    evaluation_id = response.items[0].id
    run_evaluation_response = humanloop_client.evaluations.list_runs_for_evaluation(id=evaluation_id)  # type: ignore [attr-defined]
    assert run_evaluation_response.runs[0].status == "completed"


def test_eval_run_version_id(
    get_humanloop_client: GetHumanloopClientFn,
    output_not_null_evaluator: ResourceIdentifiers,
    eval_dataset: ResourceIdentifiers,
    eval_prompt: ResourceIdentifiers,
    test_prompt_config: dict[str, Any],
) -> None:
    humanloop_client = get_humanloop_client()
    # GIVEN a prompt where a non-default version is created
    new_test_prompt_config = test_prompt_config.copy()
    new_test_prompt_config["temperature"] = 1
    new_prompt_version_response = humanloop_client.prompts.upsert(
        path=eval_prompt.file_path,
        **new_test_prompt_config,
    )
    # WHEN creating an evaluation using version_id
    humanloop_client.evaluations.run(  # type: ignore [attr-defined]
        name="test_eval_run",
        file={
            "id": new_prompt_version_response.id,
            "version_id": new_prompt_version_response.version_id,
            "type": "prompt",
        },
        dataset={
            "path": eval_dataset.file_path,
        },
        evaluators=[
            {
                "path": output_not_null_evaluator.file_path,
            }
        ],
    )
    # THEN we evaluate the version created in the test
    evaluations_response = humanloop_client.evaluations.list(file_id=new_prompt_version_response.id)
    assert evaluations_response.items and len(evaluations_response.items) == 1
    evaluation_id = evaluations_response.items[0].id
    runs_response = humanloop_client.evaluations.list_runs_for_evaluation(id=evaluation_id)
    assert runs_response.runs[0].status == "completed"
    assert (
        runs_response.runs[0].version
        and runs_response.runs[0].version.version_id == new_prompt_version_response.version_id
    )
    list_versions_response = humanloop_client.prompts.list_versions(id=new_prompt_version_response.id)
    assert list_versions_response.records and len(list_versions_response.records) == 2
    # THEN the version used in evaluation is not the default version
    response = humanloop_client.prompts.get(id=new_prompt_version_response.id)
    assert response.version_id != new_prompt_version_response.version_id


def test_eval_run_environment(
    get_humanloop_client: GetHumanloopClientFn,
    output_not_null_evaluator: ResourceIdentifiers,
    eval_dataset: ResourceIdentifiers,
    eval_prompt: ResourceIdentifiers,
    test_prompt_config: dict[str, Any],
    id_for_staging_environment: str,
) -> None:
    humanloop_client = get_humanloop_client()
    # GIVEN a prompt deployed to staging environment
    new_test_prompt_config = test_prompt_config.copy()
    new_test_prompt_config["temperature"] = 1
    new_prompt_version_response = humanloop_client.prompts.upsert(
        path=eval_prompt.file_path,
        **new_test_prompt_config,
    )
    humanloop_client.prompts.set_deployment(
        id=new_prompt_version_response.id,
        environment_id=id_for_staging_environment,
        version_id=new_prompt_version_response.version_id,
    )
    # WHEN creating an evaluation using environment
    humanloop_client.evaluations.run(  # type: ignore [attr-defined]
        name="test_eval_run",
        file={
            "id": new_prompt_version_response.id,
            "type": "prompt",
            "environment": "staging",
        },
        dataset={
            "path": eval_dataset.file_path,
        },
        evaluators=[
            {
                "path": output_not_null_evaluator.file_path,
            }
        ],
    )
    # THEN evaluation is done with the version deployed to staging environment
    evaluations_response = humanloop_client.evaluations.list(file_id=new_prompt_version_response.id)
    assert evaluations_response.items and len(evaluations_response.items) == 1
    evaluation_id = evaluations_response.items[0].id
    runs_response = humanloop_client.evaluations.list_runs_for_evaluation(id=evaluation_id)
    assert runs_response.runs[0].status == "completed"
    assert (
        runs_response.runs[0].version
        and runs_response.runs[0].version.version_id == new_prompt_version_response.version_id
    )
    default_prompt_version_response = humanloop_client.prompts.get(id=new_prompt_version_response.id)
    assert default_prompt_version_response.version_id != new_prompt_version_response.version_id


@pytest.mark.parametrize("version_lookup", ["version_id", "environment"])
def test_eval_run_version_lookup_fails_with_path(
    get_humanloop_client: GetHumanloopClientFn,
    eval_prompt: ResourceIdentifiers,
    eval_dataset: ResourceIdentifiers,
    output_not_null_evaluator: ResourceIdentifiers,
    version_lookup: str,
):
    # GIVEN an eval run where we try to evaluate a non-default version
    with pytest.raises(HumanloopRuntimeError) as e:
        humanloop_client = get_humanloop_client()
        humanloop_client.evaluations.run(  # type: ignore [attr-defined]
            name="test_eval_run",
            file={
                "path": eval_prompt.file_path,
                "type": "prompt",
                # WHEN the File id is not passed in file
                version_lookup: "will_not_work",
            },
            dataset={
                "path": eval_dataset.file_path,
            },
            evaluators=[
                {
                    "path": output_not_null_evaluator.file_path,
                }
            ],
        )
    # THEN an error is raised
    assert "You must provide the `file.id` when addressing a file by version ID or environment" in str(e.value)


def test_eval_run_with_version_upsert(
    get_humanloop_client: GetHumanloopClientFn,
    eval_prompt: ResourceIdentifiers,
    eval_dataset: ResourceIdentifiers,
    output_not_null_evaluator: ResourceIdentifiers,
    test_prompt_config: dict[str, Any],
):
    humanloop_client = get_humanloop_client()
    humanloop_client.evaluations.run(  # type: ignore [attr-defined]
        name="test_eval_run",
        file={
            "path": eval_prompt.file_path,
            "type": "prompt",
            "version": {
                **test_prompt_config,
                "temperature": 1,
            },
        },
        dataset={
            "path": eval_dataset.file_path,
        },
        evaluators=[
            {
                "path": output_not_null_evaluator.file_path,
            }
        ],
    )
    # THEN the version is upserted and evaluation finishes successfully
    evaluations_response = humanloop_client.evaluations.list(file_id=eval_prompt.file_id)
    assert evaluations_response.items and len(evaluations_response.items) == 1
    evaluation_id = evaluations_response.items[0].id
    runs_response = humanloop_client.evaluations.list_runs_for_evaluation(id=evaluation_id)
    assert runs_response.runs[0].status == "completed"
    # THEN a version was upserted based on file.version
    list_prompt_versions_response = humanloop_client.prompts.list_versions(id=eval_prompt.file_id)
    assert list_prompt_versions_response.records and len(list_prompt_versions_response.records) == 2


def test_flow_eval_does_not_work_without_callable(
    get_humanloop_client: GetHumanloopClientFn,
    eval_dataset: ResourceIdentifiers,
    output_not_null_evaluator: ResourceIdentifiers,
):
    with pytest.raises(HumanloopRuntimeError) as e:
        humanloop_client = get_humanloop_client()
        humanloop_client.evaluations.run(  # type: ignore [attr-defined]
            name="test_eval_run",
            file={
                "path": "Test Flow",
                "type": "flow",
                "version": {
                    "attributes": {
                        "foo": "bar",
                    }
                },
            },
            dataset={
                "path": eval_dataset.file_path,
            },
            evaluators=[
                {
                    "path": output_not_null_evaluator.file_path,
                }
            ],
        )
    # THEN an error is raised
    assert "You must provide a `callable` for your Flow `file` to run a local eval." in str(e.value)


def test_flow_eval_works_with_callable(
    get_humanloop_client: GetHumanloopClientFn,
    eval_dataset: ResourceIdentifiers,
    output_not_null_evaluator: ResourceIdentifiers,
    sdk_test_dir: str,
):
    humanloop_client = get_humanloop_client()
    flow_path = f"{sdk_test_dir}/Test Flow"
    # GIVEN a flow with a callable
    flow_response = humanloop_client.flows.upsert(
        path=flow_path,
        attributes={
            "foo": "bar",
        },
    )
    try:
        flow = humanloop_client.flows.upsert(
            path=flow_path,
            attributes={
                "foo": "bar",
            },
        )
        # WHEN we run an evaluation with the flow
        humanloop_client.evaluations.run(  # type: ignore [attr-defined]
            name="test_eval_run",
            file={
                "id": flow.id,
                "type": "flow",
                "callable": lambda question: "bar",
            },
            dataset={
                "path": eval_dataset.file_path,
            },
            evaluators=[
                {
                    "path": output_not_null_evaluator.file_path,
                }
            ],
        )
        # THEN the evaluation finishes successfully
        evaluations_response = humanloop_client.evaluations.list(file_id=flow.id)
        assert evaluations_response.items and len(evaluations_response.items) == 1
        evaluation_id = evaluations_response.items[0].id
        runs_response = humanloop_client.evaluations.list_runs_for_evaluation(id=evaluation_id)
        assert runs_response.runs[0].status == "completed"
    finally:
        humanloop_client.flows.delete(id=flow_response.id)


def test_cannot_evaluate_agent_with_callable(
    get_humanloop_client: GetHumanloopClientFn,
    eval_dataset: ResourceIdentifiers,
    output_not_null_evaluator: ResourceIdentifiers,
):
    with pytest.raises(ValueError) as e:
        humanloop_client = get_humanloop_client()
        humanloop_client.evaluations.run(  # type: ignore [attr-defined]
            name="test_eval_run",
            file={
                "path": "Test Agent",
                "type": "agent",
                "callable": lambda question: "bar",
            },
            dataset={
                "path": eval_dataset.file_path,
            },
            evaluators=[
                {
                    "path": output_not_null_evaluator.file_path,
                }
            ],
        )
    assert str(e.value) == "Agent evaluation is only possible on the Humanloop runtime, do not provide a `callable`."


def test_flow_eval_resolves_to_default_with_callable(
    get_humanloop_client: GetHumanloopClientFn,
    output_not_null_evaluator: ResourceIdentifiers,
    eval_dataset: ResourceIdentifiers,
    sdk_test_dir: str,
) -> None:
    humanloop_client = get_humanloop_client()
    # GIVEN a flow with some attributes
    flow_path = f"{sdk_test_dir}/Test Flow"
    flow_response = humanloop_client.flows.upsert(
        path=flow_path,
        attributes={
            "foo": "bar",
        },
    )
    try:
        # WHEN running an evaluation with the flow's callable but no version
        humanloop_client.evaluations.run(  # type: ignore [attr-defined]
            name="test_eval_run",
            file={
                "id": flow_response.id,
                "type": "flow",
                "callable": lambda question: "It's complicated don't worry about it",
            },
            dataset={
                "path": eval_dataset.file_path,
            },
            evaluators=[
                {
                    "path": output_not_null_evaluator.file_path,
                }
            ],
        )
        # THEN the evaluation finishes successfully
        evaluations_response = humanloop_client.evaluations.list(file_id=flow_response.id)
        assert evaluations_response.items and len(evaluations_response.items) == 1
        evaluation_id = evaluations_response.items and evaluations_response.items[0].id
        runs_response = humanloop_client.evaluations.list_runs_for_evaluation(id=evaluation_id)  # type: ignore [attr-defined, arg-type]
        assert runs_response.runs[0].status == "completed"
    finally:
        # Clean up test resources
        humanloop_client.flows.delete(id=flow_response.id)


@pytest.mark.skip(reason="Skip until agents are in prod")
def test_agent_eval_works_upserting(
    get_humanloop_client: GetHumanloopClientFn,
    eval_dataset: ResourceIdentifiers,
    output_not_null_evaluator: ResourceIdentifiers,
    sdk_test_dir: str,
):
    humanloop_client = get_humanloop_client()
    humanloop_client.evaluations.run(  # type: ignore [attr-defined]
        name="test_eval_run",
        file={
            "path": f"{sdk_test_dir}/Test Agent",
            "type": "agent",
            "version": {
                "model": "gpt-4o",
                "template": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant, offering very short answers.",
                    },
                    {
                        "role": "user",
                        "content": "{{question}}",
                    },
                ],
                "provider": "openai",
                "temperature": 0,
                "max_iterations": 5,
            },
        },
        dataset={
            "path": eval_dataset.file_path,
        },
        evaluators=[
            {
                "path": output_not_null_evaluator.file_path,
            }
        ],
    )
    files_response = humanloop_client.files.list_files(page=1, size=100)
    eval_agent = None
    for file in files_response.records:
        if file.path == f"{sdk_test_dir}/Test Agent":
            eval_agent = file
            break
    assert eval_agent and eval_agent.type == "agent"
    # THEN the evaluation finishes successfully
    evaluations_response = humanloop_client.evaluations.list(file_id=eval_agent.id)
    assert evaluations_response.items and len(evaluations_response.items) == 1
    evaluation_id = evaluations_response.items[0].id
    runs_response = humanloop_client.evaluations.list_runs_for_evaluation(id=evaluation_id)  # type: ignore [attr-defined, arg-type]
    assert runs_response.runs[0].status == "completed"
