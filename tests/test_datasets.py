# This file was auto-generated by Fern from our API Definition.

import typing

import pytest

from humanloop import AsyncHumanloop, Humanloop

from .utilities import validate_response


async def test_upsert(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "id": "ds_mno345",
        "path": "test-questions",
        "name": "test-questions",
        "version_id": "dsv_pqr678",
        "type": "dataset",
        "created_at": "2024-05-01T12:00:00Z",
        "updated_at": "2024-05-01T12:00:00Z",
        "status": "committed",
        "last_used_at": "2024-05-01T12:00:00Z",
        "datapoints_count": 4,
    }
    expected_types: typing.Any = {
        "id": None,
        "path": None,
        "name": None,
        "version_id": None,
        "type": None,
        "created_at": "datetime",
        "updated_at": "datetime",
        "status": None,
        "last_used_at": "datetime",
        "datapoints_count": "integer",
    }
    response = client.datasets.upsert(
        path="test-questions",
        datapoints=[
            {"inputs": {"question": "What is the capital of France?"}, "target": {"answer": "Paris"}},
            {"inputs": {"question": "Who wrote Hamlet?"}, "target": {"answer": "William Shakespeare"}},
        ],
        action="add",
        commit_message="Add two new questions and answers",
    )
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.datasets.upsert(
        path="test-questions",
        datapoints=[
            {"inputs": {"question": "What is the capital of France?"}, "target": {"answer": "Paris"}},
            {"inputs": {"question": "Who wrote Hamlet?"}, "target": {"answer": "William Shakespeare"}},
        ],
        action="add",
        commit_message="Add two new questions and answers",
    )
    validate_response(async_response, expected_response, expected_types)


async def test_get(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "id": "ds_345mno",
        "path": "test-questions",
        "name": "test-questions",
        "version_id": "dsv_678pqr",
        "type": "dataset",
        "created_at": "2024-05-01T12:00:00Z",
        "updated_at": "2024-05-01T12:00:00Z",
        "status": "committed",
        "last_used_at": "2024-05-01T12:00:00Z",
        "datapoints_count": 2,
        "datapoints": [
            {
                "id": "dp_123456",
                "inputs": {"question": "What is the capital of France?"},
                "target": {"answer": "Paris"},
            },
            {
                "id": "dp_789012",
                "inputs": {"question": "Who wrote Hamlet?"},
                "target": {"answer": "William Shakespeare"},
            },
        ],
    }
    expected_types: typing.Any = {
        "id": None,
        "path": None,
        "name": None,
        "version_id": None,
        "type": None,
        "created_at": "datetime",
        "updated_at": "datetime",
        "status": None,
        "last_used_at": "datetime",
        "datapoints_count": "integer",
        "datapoints": (
            "list",
            {
                0: {"id": None, "inputs": ("dict", {0: (None, None)}), "target": ("dict", {0: (None, None)})},
                1: {"id": None, "inputs": ("dict", {0: (None, None)}), "target": ("dict", {0: (None, None)})},
            },
        ),
    }
    response = client.datasets.get(id="ds_b0baF1ca7652", version_id="dsv_6L78pqrdFi2xa", include_datapoints=True)
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.datasets.get(
        id="ds_b0baF1ca7652", version_id="dsv_6L78pqrdFi2xa", include_datapoints=True
    )
    validate_response(async_response, expected_response, expected_types)


async def test_delete(client: Humanloop, async_client: AsyncHumanloop) -> None:
    # Type ignore to avoid mypy complaining about the function not being meant to return a value
    assert client.datasets.delete(id="id") is None  # type: ignore[func-returns-value]

    assert await async_client.datasets.delete(id="id") is None  # type: ignore[func-returns-value]


async def test_move(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "path": "path",
        "id": "id",
        "name": "name",
        "version_id": "version_id",
        "type": "dataset",
        "environments": [{"id": "id", "created_at": "2024-01-15T09:30:00Z", "name": "name", "tag": "default"}],
        "created_at": "2024-01-15T09:30:00Z",
        "updated_at": "2024-01-15T09:30:00Z",
        "created_by": {"id": "id", "email_address": "email_address", "full_name": "full_name"},
        "status": "uncommitted",
        "last_used_at": "2024-01-15T09:30:00Z",
        "commit_message": "commit_message",
        "datapoints_count": 1,
        "datapoints": [
            {"inputs": {"key": "value"}, "messages": [{"role": "user"}], "target": {"key": "value"}, "id": "id"}
        ],
    }
    expected_types: typing.Any = {
        "path": None,
        "id": None,
        "name": None,
        "version_id": None,
        "type": None,
        "environments": ("list", {0: {"id": None, "created_at": "datetime", "name": None, "tag": None}}),
        "created_at": "datetime",
        "updated_at": "datetime",
        "created_by": {"id": None, "email_address": None, "full_name": None},
        "status": None,
        "last_used_at": "datetime",
        "commit_message": None,
        "datapoints_count": "integer",
        "datapoints": (
            "list",
            {
                0: {
                    "inputs": ("dict", {0: (None, None)}),
                    "messages": ("list", {0: {"role": None}}),
                    "target": ("dict", {0: (None, None)}),
                    "id": None,
                }
            },
        ),
    }
    response = client.datasets.move(id="id")
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.datasets.move(id="id")
    validate_response(async_response, expected_response, expected_types)


async def test_list_versions(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "records": [
            {
                "id": "ds_345mno",
                "path": "test-questions",
                "name": "test-questions",
                "version_id": "dsv_678pqr",
                "type": "dataset",
                "created_at": "2024-05-01T12:00:00Z",
                "updated_at": "2024-05-01T12:00:00Z",
                "created_by": {
                    "id": "usr_v23rSVAgas2a",
                    "full_name": "Jordan Burges",
                    "email_address": "jordan@humanloop.com",
                },
                "status": "committed",
                "commit_message": "initial commit",
                "last_used_at": "2024-05-01T12:00:00Z",
                "datapoints_count": 2,
            }
        ]
    }
    expected_types: typing.Any = {
        "records": (
            "list",
            {
                0: {
                    "id": None,
                    "path": None,
                    "name": None,
                    "version_id": None,
                    "type": None,
                    "created_at": "datetime",
                    "updated_at": "datetime",
                    "created_by": {"id": None, "full_name": None, "email_address": None},
                    "status": None,
                    "commit_message": None,
                    "last_used_at": "datetime",
                    "datapoints_count": "integer",
                }
            },
        )
    }
    response = client.datasets.list_versions(id="ds_b0baF1ca7652", status="committed")
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.datasets.list_versions(id="ds_b0baF1ca7652", status="committed")
    validate_response(async_response, expected_response, expected_types)


async def test_commit(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "id": "ds_345mno",
        "path": "test-questions",
        "name": "test-questions",
        "version_id": "dsv_678pqr",
        "type": "dataset",
        "created_at": "2024-05-01T12:00:00Z",
        "updated_at": "2024-05-01T12:00:00Z",
        "created_by": {"id": "usr_v23rSVAgas2a", "full_name": "Jordan Burges", "email_address": "jordan@humanloop.com"},
        "status": "committed",
        "commit_message": "initial commit",
        "last_used_at": "2024-05-01T12:00:00Z",
        "datapoints_count": 2,
    }
    expected_types: typing.Any = {
        "id": None,
        "path": None,
        "name": None,
        "version_id": None,
        "type": None,
        "created_at": "datetime",
        "updated_at": "datetime",
        "created_by": {"id": None, "full_name": None, "email_address": None},
        "status": None,
        "commit_message": None,
        "last_used_at": "datetime",
        "datapoints_count": "integer",
    }
    response = client.datasets.commit(
        id="ds_b0baF1ca7652", version_id="dsv_6L78pqrdFi2xa", commit_message="initial commit"
    )
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.datasets.commit(
        id="ds_b0baF1ca7652", version_id="dsv_6L78pqrdFi2xa", commit_message="initial commit"
    )
    validate_response(async_response, expected_response, expected_types)


async def test_set_deployment(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "id": "ds_345mno",
        "path": "test-questions",
        "name": "test-questions",
        "version_id": "dsv_678pqr",
        "type": "dataset",
        "created_at": "2024-05-01T12:00:00Z",
        "updated_at": "2024-05-01T12:00:00Z",
        "created_by": {"id": "usr_v23rSVAgas2a", "full_name": "Jordan Burges", "email_address": "jordan@humanloop.com"},
        "status": "committed",
        "commit_message": "initial commit",
        "last_used_at": "2024-05-01T12:00:00Z",
        "datapoints_count": 2,
    }
    expected_types: typing.Any = {
        "id": None,
        "path": None,
        "name": None,
        "version_id": None,
        "type": None,
        "created_at": "datetime",
        "updated_at": "datetime",
        "created_by": {"id": None, "full_name": None, "email_address": None},
        "status": None,
        "commit_message": None,
        "last_used_at": "datetime",
        "datapoints_count": "integer",
    }
    response = client.datasets.set_deployment(
        id="ds_b0baF1ca7652", environment_id="staging", version_id="dsv_6L78pqrdFi2xa"
    )
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.datasets.set_deployment(
        id="ds_b0baF1ca7652", environment_id="staging", version_id="dsv_6L78pqrdFi2xa"
    )
    validate_response(async_response, expected_response, expected_types)


async def test_remove_deployment(client: Humanloop, async_client: AsyncHumanloop) -> None:
    # Type ignore to avoid mypy complaining about the function not being meant to return a value
    assert client.datasets.remove_deployment(id="ds_b0baF1ca7652", environment_id="staging") is None  # type: ignore[func-returns-value]

    assert await async_client.datasets.remove_deployment(id="ds_b0baF1ca7652", environment_id="staging") is None  # type: ignore[func-returns-value]


@pytest.mark.skip(reason="Untested")
async def test_list_environments(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = [
        {
            "id": "id",
            "created_at": "2024-01-15T09:30:00Z",
            "name": "name",
            "tag": "default",
            "file": {
                "path": "path",
                "id": "id",
                "name": "name",
                "version_id": "version_id",
                "type": "prompt",
                "environments": [{"id": "id", "created_at": "2024-01-15T09:30:00Z", "name": "name", "tag": "default"}],
                "created_at": "2024-01-15T09:30:00Z",
                "updated_at": "2024-01-15T09:30:00Z",
                "created_by": {"id": "id", "email_address": "email_address"},
                "status": "uncommitted",
                "last_used_at": "2024-01-15T09:30:00Z",
                "model": "model",
                "endpoint": "complete",
                "template": "template",
                "provider": "openai",
                "max_tokens": 1,
                "temperature": 1.1,
                "top_p": 1.1,
                "stop": "stop",
                "presence_penalty": 1.1,
                "frequency_penalty": 1.1,
                "other": {"key": "value"},
                "seed": 1,
                "response_format": {"type": "json_object"},
                "tools": [{"name": "name", "description": "description"}],
                "linked_tools": [
                    {"name": "name", "description": "description", "id": "id", "version_id": "version_id"}
                ],
                "commit_message": "commit_message",
                "version_logs_count": 1,
                "total_logs_count": 1,
                "inputs": [{"name": "name"}],
                "evaluator_aggregates": [
                    {
                        "value": 1.1,
                        "evaluator_id": "evaluator_id",
                        "evaluator_version_id": "evaluator_version_id",
                        "created_at": "2024-01-15T09:30:00Z",
                        "updated_at": "2024-01-15T09:30:00Z",
                    }
                ],
            },
        }
    ]
    expected_types: typing.Tuple[typing.Any, typing.Any] = (
        "list",
        {
            0: {
                "id": None,
                "created_at": "datetime",
                "name": None,
                "tag": None,
                "file": {
                    "path": None,
                    "id": None,
                    "name": None,
                    "version_id": None,
                    "type": None,
                    "environments": ("list", {0: {"id": None, "created_at": "datetime", "name": None, "tag": None}}),
                    "created_at": "datetime",
                    "updated_at": "datetime",
                    "created_by": {"id": None, "email_address": None},
                    "status": None,
                    "last_used_at": "datetime",
                    "model": None,
                    "endpoint": None,
                    "template": None,
                    "provider": None,
                    "max_tokens": "integer",
                    "temperature": None,
                    "top_p": None,
                    "stop": None,
                    "presence_penalty": None,
                    "frequency_penalty": None,
                    "other": ("dict", {0: (None, None)}),
                    "seed": "integer",
                    "response_format": {"type": None},
                    "tools": ("list", {0: {"name": None, "description": None}}),
                    "linked_tools": ("list", {0: {"name": None, "description": None, "id": None, "version_id": None}}),
                    "commit_message": None,
                    "version_logs_count": "integer",
                    "total_logs_count": "integer",
                    "inputs": ("list", {0: {"name": None}}),
                    "evaluator_aggregates": (
                        "list",
                        {
                            0: {
                                "value": None,
                                "evaluator_id": None,
                                "evaluator_version_id": None,
                                "created_at": "datetime",
                                "updated_at": "datetime",
                            }
                        },
                    ),
                },
            }
        },
    )
    response = client.datasets.list_environments(id="id")
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.datasets.list_environments(id="id")
    validate_response(async_response, expected_response, expected_types)
