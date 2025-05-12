import time
from typing import Any

from openai import OpenAI
from humanloop.client import Humanloop


def test_prompt_decorator(
    humanloop_test_client: Humanloop,
    sdk_test_dir: str,
    test_prompt_config: dict[str, Any],
    openai_key: str,
):
    try:
        prompt_path = f"{sdk_test_dir}/test_prompt"
        prompt_response = humanloop_test_client.prompts.upsert(
            path=prompt_path,
            **test_prompt_config,
        )

        prompt_versions_response = humanloop_test_client.prompts.list_versions(id=prompt_response.id)
        assert len(prompt_versions_response.records) == 1

        @humanloop_test_client.prompt(path=prompt_path)
        def my_prompt(question: str) -> str:
            openai_client = OpenAI(api_key=openai_key)

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": question}],
            )

            assert response.choices[0].message.content is not None
            return response.choices[0].message.content

        assert "paris" in my_prompt("What is the capital of the France?").lower()

        time.sleep(5)
        prompt_versions_response = humanloop_test_client.prompts.list_versions(id=prompt_response.id)
        assert len(prompt_versions_response.records) == 2

        logs_response = humanloop_test_client.logs.list(file_id=prompt_response.id, page=1, size=50)

        assert logs_response.items is not None and len(logs_response.items) == 1
    finally:
        humanloop_test_client.prompts.delete(id=prompt_response.id)


def test_call_prompt_in_flow_decorator(
    humanloop_test_client: Humanloop,
    sdk_test_dir: str,
    openai_key: str,
):
    try:

        @humanloop_test_client.flow(path=f"{sdk_test_dir}/test_flow")
        def my_flow(question: str) -> str:
            response = humanloop_test_client.prompts.call(
                path=f"{sdk_test_dir}/test_prompt",
                prompt={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "temperature": 0,
                },
                messages=[{"role": "user", "content": question}],
                provider_api_keys={"openai": openai_key},
            )

            assert response.logs[0].output is not None
            return response.logs[0].output

        assert "paris" in my_flow("What is the capital of the France?").lower()
        time.sleep(5)
        prompt_response = humanloop_test_client.files.retrieve_by_path(path=f"{sdk_test_dir}/test_prompt")
        assert prompt_response is not None
        prompt_logs_response = humanloop_test_client.logs.list(file_id=prompt_response.id, page=1, size=50)
        assert prompt_logs_response.items is not None and len(prompt_logs_response.items) == 1
        prompt_log = prompt_logs_response.items[0]

        flow_response = humanloop_test_client.files.retrieve_by_path(path=f"{sdk_test_dir}/test_flow")
        assert flow_response is not None
        flow_logs_response = humanloop_test_client.logs.list(file_id=flow_response.id, page=1, size=50)
        assert flow_logs_response.items is not None and len(flow_logs_response.items) == 1
        flow_log = flow_logs_response.items[0]
        assert prompt_log.trace_parent_id == flow_log.id
    finally:
        flow_response = humanloop_test_client.files.retrieve_by_path(path=f"{sdk_test_dir}/test_flow")
        if flow_response is not None:
            humanloop_test_client.flows.delete(id=flow_response.id)
        prompt_response = humanloop_test_client.files.retrieve_by_path(path=f"{sdk_test_dir}/test_prompt")
        if prompt_response is not None:
            humanloop_test_client.prompts.delete(id=prompt_response.id)


def test_flow_decorator_logs_exceptions(
    humanloop_test_client: Humanloop,
    sdk_test_dir: str,
):
    try:

        @humanloop_test_client.flow(path=f"{sdk_test_dir}/test_flow_log_error")
        def my_flow(question: str) -> str:
            raise ValueError("This is a test exception")

        my_flow("test")

        time.sleep(5)

        flow_response = humanloop_test_client.files.retrieve_by_path(path=f"{sdk_test_dir}/test_flow_log_error")
        assert flow_response is not None
        flow_logs_response = humanloop_test_client.logs.list(file_id=flow_response.id, page=1, size=50)
        assert flow_logs_response.items is not None and len(flow_logs_response.items) == 1
        flow_log = flow_logs_response.items[0]
        assert flow_log.error is not None
        assert flow_log.output is None

    finally:
        flow_response = humanloop_test_client.files.retrieve_by_path(path=f"{sdk_test_dir}/test_flow_log_error")
        if flow_response is not None:
            humanloop_test_client.flows.delete(id=flow_response.id)


def test_flow_decorator_populates_output_message(
    humanloop_test_client: Humanloop,
    sdk_test_dir: str,
):
    try:

        @humanloop_test_client.flow(path=f"{sdk_test_dir}/test_flow_log_output_message")
        def my_flow(question: str) -> dict[str, Any]:
            return {"role": "user", "content": question}

        assert "france" in my_flow("What is the capital of the France?")["content"].lower()

        time.sleep(5)

        flow_response = humanloop_test_client.files.retrieve_by_path(
            path=f"{sdk_test_dir}/test_flow_log_output_message"
        )
        assert flow_response is not None
        flow_logs_response = humanloop_test_client.logs.list(file_id=flow_response.id, page=1, size=50)
        assert flow_logs_response.items is not None and len(flow_logs_response.items) == 1
        flow_log = flow_logs_response.items[0]
        assert flow_log.output_message is not None
        assert flow_log.output is None
        assert flow_log.error is None

    finally:
        flow_response = humanloop_test_client.files.retrieve_by_path(
            path=f"{sdk_test_dir}/test_flow_log_output_message"
        )
        if flow_response is not None:
            humanloop_test_client.flows.delete(id=flow_response.id)
