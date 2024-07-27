# This file was auto-generated by Fern from our API Definition.

import typing

import pytest

from humanloop import AsyncHumanloop, Humanloop

from .utilities import validate_response


async def test_log(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "id": "data_abc123",
        "tool_id": "tl_def456",
        "version_id": "tv_ghi789",
        "session_id": "sesh_hw012",
    }
    expected_types: typing.Any = {"id": None, "tool_id": None, "version_id": None, "session_id": None}
    response = client.tools.log(
        path="math-tool",
        tool={
            "function": {
                "name": "multiply",
                "description": "Multiply two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            }
        },
        inputs={"a": 5, "b": 7},
        output="35",
    )
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.tools.log(
        path="math-tool",
        tool={
            "function": {
                "name": "multiply",
                "description": "Multiply two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            }
        },
        inputs={"a": 5, "b": 7},
        output="35",
    )
    validate_response(async_response, expected_response, expected_types)


async def test_upsert(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "id": "tl_789ghi",
        "path": "math-tool",
        "name": "math-tool",
        "version_id": "tv_012jkl",
        "type": "tool",
        "created_at": "2024-05-01T12:00:00Z",
        "updated_at": "2024-05-01T12:00:00Z",
        "status": "committed",
        "last_used_at": "2024-05-01T12:00:00Z",
        "version_logs_count": 1,
        "total_logs_count": 1,
        "inputs": [{"name": "operation"}],
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
        "version_logs_count": "integer",
        "total_logs_count": "integer",
        "inputs": ("list", {0: {"name": None}}),
    }
    response = client.tools.upsert(
        path="math-tool",
        function={
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
        commit_message="Initial commit",
    )
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.tools.upsert(
        path="math-tool",
        function={
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
        commit_message="Initial commit",
    )
    validate_response(async_response, expected_response, expected_types)


async def test_get(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "id": "tl_789ghi",
        "path": "math-tool",
        "name": "math-tool",
        "version_id": "tv_012jkl",
        "type": "tool",
        "created_at": "2024-05-01T12:00:00Z",
        "updated_at": "2024-05-01T12:00:00Z",
        "status": "committed",
        "last_used_at": "2024-05-01T12:00:00Z",
        "version_logs_count": 1,
        "total_logs_count": 1,
        "inputs": [{"name": "operation"}],
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
        "version_logs_count": "integer",
        "total_logs_count": "integer",
        "inputs": ("list", {0: {"name": None}}),
    }
    response = client.tools.get(id="tl_789ghi")
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.tools.get(id="tl_789ghi")
    validate_response(async_response, expected_response, expected_types)


async def test_delete(client: Humanloop, async_client: AsyncHumanloop) -> None:
    # Type ignore to avoid mypy complaining about the function not being meant to return a value
    assert client.tools.delete(id="tl_789ghi") is None  # type: ignore[func-returns-value]

    assert await async_client.tools.delete(id="tl_789ghi") is None  # type: ignore[func-returns-value]


async def test_move(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "id": "tl_789ghi",
        "path": "math-tool",
        "name": "math-tool",
        "version_id": "tv_012jkl",
        "type": "tool",
        "created_at": "2024-05-01T12:00:00Z",
        "updated_at": "2024-05-01T12:00:00Z",
        "status": "committed",
        "last_used_at": "2024-05-01T12:00:00Z",
        "version_logs_count": 1,
        "total_logs_count": 1,
        "inputs": [{"name": "operation"}],
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
        "version_logs_count": "integer",
        "total_logs_count": "integer",
        "inputs": ("list", {0: {"name": None}}),
    }
    response = client.tools.move(id="tl_789ghi", path="new directory/new name")
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.tools.move(id="tl_789ghi", path="new directory/new name")
    validate_response(async_response, expected_response, expected_types)


async def test_list_versions(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "records": [
            {
                "id": "tl_789ghi",
                "path": "math-tool",
                "name": "math-tool",
                "version_id": "tv_012jkl",
                "type": "tool",
                "created_at": "2024-05-01T12:00:00Z",
                "updated_at": "2024-05-01T12:00:00Z",
                "status": "committed",
                "last_used_at": "2024-05-01T12:00:00Z",
                "version_logs_count": 1,
                "total_logs_count": 1,
                "inputs": [{"name": "operation"}],
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
                    "status": None,
                    "last_used_at": "datetime",
                    "version_logs_count": "integer",
                    "total_logs_count": "integer",
                    "inputs": ("list", {0: {"name": None}}),
                }
            },
        )
    }
    response = client.tools.list_versions(id="tl_789ghi", status="committed")
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.tools.list_versions(id="tl_789ghi", status="committed")
    validate_response(async_response, expected_response, expected_types)


async def test_commit(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "id": "tl_789ghi",
        "path": "math-tool",
        "name": "math-tool",
        "version_id": "tv_012jkl",
        "type": "tool",
        "created_at": "2024-05-01T12:00:00Z",
        "updated_at": "2024-05-01T12:00:00Z",
        "status": "committed",
        "last_used_at": "2024-05-01T12:00:00Z",
        "version_logs_count": 1,
        "total_logs_count": 1,
        "inputs": [{"name": "operation"}],
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
        "version_logs_count": "integer",
        "total_logs_count": "integer",
        "inputs": ("list", {0: {"name": None}}),
    }
    response = client.tools.commit(id="tl_789ghi", version_id="tv_012jkl", commit_message="Initial commit")
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.tools.commit(
        id="tl_789ghi", version_id="tv_012jkl", commit_message="Initial commit"
    )
    validate_response(async_response, expected_response, expected_types)


async def test_update_monitoring(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "id": "tl_789ghi",
        "path": "math-tool",
        "name": "math-tool",
        "version_id": "tv_012jkl",
        "type": "tool",
        "created_at": "2024-05-01T12:00:00Z",
        "updated_at": "2024-05-01T12:00:00Z",
        "status": "committed",
        "last_used_at": "2024-05-01T12:00:00Z",
        "version_logs_count": 1,
        "total_logs_count": 1,
        "inputs": [{"name": "operation"}],
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
        "version_logs_count": "integer",
        "total_logs_count": "integer",
        "inputs": ("list", {0: {"name": None}}),
    }
    response = client.tools.update_monitoring(id="tl_789ghi", activate=[{"evaluator_version_id": "evv_1abc4308abd"}])
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.tools.update_monitoring(
        id="tl_789ghi", activate=[{"evaluator_version_id": "evv_1abc4308abd"}]
    )
    validate_response(async_response, expected_response, expected_types)


async def test_set_deployment(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "id": "tl_789ghi",
        "path": "math-tool",
        "name": "math-tool",
        "version_id": "tv_012jkl",
        "type": "tool",
        "created_at": "2024-05-01T12:00:00Z",
        "updated_at": "2024-05-01T12:00:00Z",
        "status": "committed",
        "last_used_at": "2024-05-01T12:00:00Z",
        "version_logs_count": 1,
        "total_logs_count": 1,
        "inputs": [{"name": "operation"}],
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
        "version_logs_count": "integer",
        "total_logs_count": "integer",
        "inputs": ("list", {0: {"name": None}}),
    }
    response = client.tools.set_deployment(id="tl_789ghi", environment_id="staging", version_id="tv_012jkl")
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.tools.set_deployment(
        id="tl_789ghi", environment_id="staging", version_id="tv_012jkl"
    )
    validate_response(async_response, expected_response, expected_types)


async def test_remove_deployment(client: Humanloop, async_client: AsyncHumanloop) -> None:
    # Type ignore to avoid mypy complaining about the function not being meant to return a value
    assert client.tools.remove_deployment(id="tl_789ghi", environment_id="staging") is None  # type: ignore[func-returns-value]

    assert await async_client.tools.remove_deployment(id="tl_789ghi", environment_id="staging") is None  # type: ignore[func-returns-value]


@pytest.mark.skip(reason="Untested")
async def test_list_environments(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = [
        {
            "id": "tl_789ghi",
            "created_at": "2024-05-01T12:00:00Z",
            "name": "production",
            "tag": "default",
            "file": {
                "id": "tl_789ghi",
                "path": "math-tool",
                "name": "math-tool",
                "version_id": "tv_012jkl",
                "type": "tool",
                "created_at": "2024-05-01T12:00:00Z",
                "updated_at": "2024-05-01T12:00:00Z",
                "status": "committed",
                "last_used_at": "2024-05-01T12:00:00Z",
                "version_logs_count": 1,
                "total_logs_count": 1,
                "inputs": [{"name": "operation"}],
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
                    "id": None,
                    "path": None,
                    "name": None,
                    "version_id": None,
                    "type": None,
                    "created_at": "datetime",
                    "updated_at": "datetime",
                    "status": None,
                    "last_used_at": "datetime",
                    "version_logs_count": "integer",
                    "total_logs_count": "integer",
                    "inputs": ("list", {0: {"name": None}}),
                },
            }
        },
    )
    response = client.tools.list_environments(id="tl_789ghi")
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.tools.list_environments(id="tl_789ghi")
    validate_response(async_response, expected_response, expected_types)
