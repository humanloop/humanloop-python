# This file was auto-generated by Fern from our API Definition.

import typing

from humanloop import AsyncHumanloop, EvaluateeRequest, EvaluationsDatasetRequest, EvaluationsRequest, Humanloop

from .utilities import validate_response


async def test_create(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "id": "ev_567yza",
        "created_at": "2024-05-01T12:00:00Z",
        "updated_at": "2024-05-01T12:00:00Z",
        "status": "completed",
        "dataset": {
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
        },
        "evaluatees": [
            {
                "version": {
                    "path": "Personal Projects/Coding Assistant",
                    "id": "pr_30gco7dx6JDq4200GVOHa",
                    "name": "Coding Assistant",
                    "version_id": "prv_7ZlQREDScH0xkhUwtXruN",
                    "type": "prompt",
                    "environments": [
                        {
                            "id": "env_ffSVxEBzJcBZ1H5jcNMVj",
                            "created_at": "2023-06-27T23:16:07.992339",
                            "name": "development",
                            "tag": "default",
                        }
                    ],
                    "created_at": "2024-07-08T22:40:35.656915",
                    "updated_at": "2024-07-08T22:40:35.656915",
                    "created_by": {
                        "id": "usr_01RJO1k2spBVqNUt1ASef",
                        "email_address": "raza@humanloop.com",
                        "full_name": "Raza Habib",
                    },
                    "status": "committed",
                    "last_used_at": "2024-07-08T22:40:35.656915",
                    "model": "gpt-4o",
                    "endpoint": "chat",
                    "template": [
                        {"content": "You are a helpful coding assistant specialising in {{language}}", "role": "system"}
                    ],
                    "provider": "openai",
                    "max_tokens": -1,
                    "temperature": 0.7,
                    "top_p": 1,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "other": {},
                    "tools": [],
                    "linked_tools": [],
                    "commit_message": "Initial commit",
                    "version_logs_count": 0,
                    "total_logs_count": 0,
                    "inputs": [{"name": "messages"}],
                },
                "orchestrated": False,
            }
        ],
        "evaluators": [
            {
                "version": {
                    "id": "ev_890bcd",
                    "name": "Accuracy Evaluator",
                    "path": "Shared Evaluators/Accuracy Evaluator",
                    "version_id": "evv_012def",
                    "type": "evaluator",
                    "created_at": "2024-05-01T12:00:00Z",
                    "updated_at": "2024-05-01T12:00:00Z",
                    "status": "committed",
                    "last_used_at": "2024-05-01T12:00:00Z",
                    "spec": {
                        "arguments_type": "target_required",
                        "return_type": "number",
                        "evaluator_type": "python",
                        "code": "def evaluate(answer, target):\\n    return 0.5",
                    },
                    "version_logs_count": 1,
                    "total_logs_count": 1,
                    "inputs": [{"name": "answer"}],
                },
                "orchestrated": False,
            }
        ],
    }
    expected_types: typing.Any = {
        "id": None,
        "created_at": "datetime",
        "updated_at": "datetime",
        "status": None,
        "dataset": {
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
        },
        "evaluatees": (
            "list",
            {
                0: {
                    "version": {
                        "path": None,
                        "id": None,
                        "name": None,
                        "version_id": None,
                        "type": None,
                        "environments": (
                            "list",
                            {0: {"id": None, "created_at": "datetime", "name": None, "tag": None}},
                        ),
                        "created_at": "datetime",
                        "updated_at": "datetime",
                        "created_by": {"id": None, "email_address": None, "full_name": None},
                        "status": None,
                        "last_used_at": "datetime",
                        "model": None,
                        "endpoint": None,
                        "template": ("list", {0: {"content": None, "role": None}}),
                        "provider": None,
                        "max_tokens": "integer",
                        "temperature": None,
                        "top_p": None,
                        "presence_penalty": None,
                        "frequency_penalty": None,
                        "other": ("dict", {}),
                        "tools": ("list", {}),
                        "linked_tools": ("list", {}),
                        "commit_message": None,
                        "version_logs_count": "integer",
                        "total_logs_count": "integer",
                        "inputs": ("list", {0: {"name": None}}),
                    },
                    "orchestrated": None,
                }
            },
        ),
        "evaluators": (
            "list",
            {
                0: {
                    "version": {
                        "id": None,
                        "name": None,
                        "path": None,
                        "version_id": None,
                        "type": None,
                        "created_at": "datetime",
                        "updated_at": "datetime",
                        "status": None,
                        "last_used_at": "datetime",
                        "spec": {"arguments_type": None, "return_type": None, "evaluator_type": None, "code": None},
                        "version_logs_count": "integer",
                        "total_logs_count": "integer",
                        "inputs": ("list", {0: {"name": None}}),
                    },
                    "orchestrated": None,
                }
            },
        ),
    }
    response = client.evaluations.create(
        dataset=EvaluationsDatasetRequest(version_id="dsv_6L78pqrdFi2xa"),
        evaluatees=[EvaluateeRequest(version_id="prv_7ZlQREDScH0xkhUwtXruN", orchestrated=False)],
        evaluators=[EvaluationsRequest(version_id="evv_012def", orchestrated=False)],
    )
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.evaluations.create(
        dataset=EvaluationsDatasetRequest(version_id="dsv_6L78pqrdFi2xa"),
        evaluatees=[EvaluateeRequest(version_id="prv_7ZlQREDScH0xkhUwtXruN", orchestrated=False)],
        evaluators=[EvaluationsRequest(version_id="evv_012def", orchestrated=False)],
    )
    validate_response(async_response, expected_response, expected_types)


async def test_get(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "id": "ev_567yza",
        "created_at": "2024-05-01T12:00:00Z",
        "updated_at": "2024-05-01T12:00:00Z",
        "status": "completed",
        "dataset": {
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
        },
        "evaluatees": [
            {
                "version": {
                    "path": "Personal Projects/Coding Assistant",
                    "id": "pr_30gco7dx6JDq4200GVOHa",
                    "name": "Coding Assistant",
                    "version_id": "prv_7ZlQREDScH0xkhUwtXruN",
                    "type": "prompt",
                    "environments": [
                        {
                            "id": "env_ffSVxEBzJcBZ1H5jcNMVj",
                            "created_at": "2023-06-27T23:16:07.992339",
                            "name": "development",
                            "tag": "default",
                        }
                    ],
                    "created_at": "2024-07-08T22:40:35.656915",
                    "updated_at": "2024-07-08T22:40:35.656915",
                    "created_by": {
                        "id": "usr_01RJO1k2spBVqNUt1ASef",
                        "email_address": "raza@humanloop.com",
                        "full_name": "Raza Habib",
                    },
                    "status": "committed",
                    "last_used_at": "2024-07-08T22:40:35.656915",
                    "model": "gpt-4o",
                    "endpoint": "chat",
                    "template": [
                        {"content": "You are a helpful coding assistant specialising in {{language}}", "role": "system"}
                    ],
                    "provider": "openai",
                    "max_tokens": -1,
                    "temperature": 0.7,
                    "top_p": 1,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "other": {},
                    "tools": [],
                    "linked_tools": [],
                    "commit_message": "Initial commit",
                    "version_logs_count": 0,
                    "total_logs_count": 0,
                    "inputs": [{"name": "messages"}],
                },
                "orchestrated": False,
            }
        ],
        "evaluators": [
            {
                "version": {
                    "id": "ev_890bcd",
                    "name": "Accuracy Evaluator",
                    "path": "Shared Evaluators/Accuracy Evaluator",
                    "version_id": "evv_012def",
                    "type": "evaluator",
                    "created_at": "2024-05-01T12:00:00Z",
                    "updated_at": "2024-05-01T12:00:00Z",
                    "status": "committed",
                    "last_used_at": "2024-05-01T12:00:00Z",
                    "spec": {
                        "arguments_type": "target_required",
                        "return_type": "number",
                        "evaluator_type": "python",
                        "code": "def evaluate(answer, target):\\n    return 0.5",
                    },
                    "version_logs_count": 1,
                    "total_logs_count": 1,
                    "inputs": [{"name": "answer"}],
                },
                "orchestrated": False,
            }
        ],
    }
    expected_types: typing.Any = {
        "id": None,
        "created_at": "datetime",
        "updated_at": "datetime",
        "status": None,
        "dataset": {
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
        },
        "evaluatees": (
            "list",
            {
                0: {
                    "version": {
                        "path": None,
                        "id": None,
                        "name": None,
                        "version_id": None,
                        "type": None,
                        "environments": (
                            "list",
                            {0: {"id": None, "created_at": "datetime", "name": None, "tag": None}},
                        ),
                        "created_at": "datetime",
                        "updated_at": "datetime",
                        "created_by": {"id": None, "email_address": None, "full_name": None},
                        "status": None,
                        "last_used_at": "datetime",
                        "model": None,
                        "endpoint": None,
                        "template": ("list", {0: {"content": None, "role": None}}),
                        "provider": None,
                        "max_tokens": "integer",
                        "temperature": None,
                        "top_p": None,
                        "presence_penalty": None,
                        "frequency_penalty": None,
                        "other": ("dict", {}),
                        "tools": ("list", {}),
                        "linked_tools": ("list", {}),
                        "commit_message": None,
                        "version_logs_count": "integer",
                        "total_logs_count": "integer",
                        "inputs": ("list", {0: {"name": None}}),
                    },
                    "orchestrated": None,
                }
            },
        ),
        "evaluators": (
            "list",
            {
                0: {
                    "version": {
                        "id": None,
                        "name": None,
                        "path": None,
                        "version_id": None,
                        "type": None,
                        "created_at": "datetime",
                        "updated_at": "datetime",
                        "status": None,
                        "last_used_at": "datetime",
                        "spec": {"arguments_type": None, "return_type": None, "evaluator_type": None, "code": None},
                        "version_logs_count": "integer",
                        "total_logs_count": "integer",
                        "inputs": ("list", {0: {"name": None}}),
                    },
                    "orchestrated": None,
                }
            },
        ),
    }
    response = client.evaluations.get(id="ev_567yza")
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.evaluations.get(id="ev_567yza")
    validate_response(async_response, expected_response, expected_types)


async def test_delete(client: Humanloop, async_client: AsyncHumanloop) -> None:
    # Type ignore to avoid mypy complaining about the function not being meant to return a value
    assert client.evaluations.delete(id="ev_567yza") is None  # type: ignore[func-returns-value]

    assert await async_client.evaluations.delete(id="ev_567yza") is None  # type: ignore[func-returns-value]


async def test_update_setup(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "id": "ev_567yza",
        "created_at": "2024-05-01T12:00:00Z",
        "updated_at": "2024-05-01T12:00:00Z",
        "status": "completed",
        "dataset": {
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
        },
        "evaluatees": [
            {
                "version": {
                    "path": "Personal Projects/Coding Assistant",
                    "id": "pr_30gco7dx6JDq4200GVOHa",
                    "name": "Coding Assistant",
                    "version_id": "prv_7ZlQREDScH0xkhUwtXruN",
                    "type": "prompt",
                    "environments": [
                        {
                            "id": "env_ffSVxEBzJcBZ1H5jcNMVj",
                            "created_at": "2023-06-27T23:16:07.992339",
                            "name": "development",
                            "tag": "default",
                        }
                    ],
                    "created_at": "2024-07-08T22:40:35.656915",
                    "updated_at": "2024-07-08T22:40:35.656915",
                    "created_by": {
                        "id": "usr_01RJO1k2spBVqNUt1ASef",
                        "email_address": "raza@humanloop.com",
                        "full_name": "Raza Habib",
                    },
                    "status": "committed",
                    "last_used_at": "2024-07-08T22:40:35.656915",
                    "model": "gpt-4o",
                    "endpoint": "chat",
                    "template": [
                        {"content": "You are a helpful coding assistant specialising in {{language}}", "role": "system"}
                    ],
                    "provider": "openai",
                    "max_tokens": -1,
                    "temperature": 0.7,
                    "top_p": 1,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "other": {},
                    "tools": [],
                    "linked_tools": [],
                    "commit_message": "Initial commit",
                    "version_logs_count": 0,
                    "total_logs_count": 0,
                    "inputs": [{"name": "messages"}],
                },
                "orchestrated": False,
            }
        ],
        "evaluators": [
            {
                "version": {
                    "id": "ev_890bcd",
                    "name": "Accuracy Evaluator",
                    "path": "Shared Evaluators/Accuracy Evaluator",
                    "version_id": "evv_012def",
                    "type": "evaluator",
                    "created_at": "2024-05-01T12:00:00Z",
                    "updated_at": "2024-05-01T12:00:00Z",
                    "status": "committed",
                    "last_used_at": "2024-05-01T12:00:00Z",
                    "spec": {
                        "arguments_type": "target_required",
                        "return_type": "number",
                        "evaluator_type": "python",
                        "code": "def evaluate(answer, target):\\n    return 0.5",
                    },
                    "version_logs_count": 1,
                    "total_logs_count": 1,
                    "inputs": [{"name": "answer"}],
                },
                "orchestrated": False,
            }
        ],
    }
    expected_types: typing.Any = {
        "id": None,
        "created_at": "datetime",
        "updated_at": "datetime",
        "status": None,
        "dataset": {
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
        },
        "evaluatees": (
            "list",
            {
                0: {
                    "version": {
                        "path": None,
                        "id": None,
                        "name": None,
                        "version_id": None,
                        "type": None,
                        "environments": (
                            "list",
                            {0: {"id": None, "created_at": "datetime", "name": None, "tag": None}},
                        ),
                        "created_at": "datetime",
                        "updated_at": "datetime",
                        "created_by": {"id": None, "email_address": None, "full_name": None},
                        "status": None,
                        "last_used_at": "datetime",
                        "model": None,
                        "endpoint": None,
                        "template": ("list", {0: {"content": None, "role": None}}),
                        "provider": None,
                        "max_tokens": "integer",
                        "temperature": None,
                        "top_p": None,
                        "presence_penalty": None,
                        "frequency_penalty": None,
                        "other": ("dict", {}),
                        "tools": ("list", {}),
                        "linked_tools": ("list", {}),
                        "commit_message": None,
                        "version_logs_count": "integer",
                        "total_logs_count": "integer",
                        "inputs": ("list", {0: {"name": None}}),
                    },
                    "orchestrated": None,
                }
            },
        ),
        "evaluators": (
            "list",
            {
                0: {
                    "version": {
                        "id": None,
                        "name": None,
                        "path": None,
                        "version_id": None,
                        "type": None,
                        "created_at": "datetime",
                        "updated_at": "datetime",
                        "status": None,
                        "last_used_at": "datetime",
                        "spec": {"arguments_type": None, "return_type": None, "evaluator_type": None, "code": None},
                        "version_logs_count": "integer",
                        "total_logs_count": "integer",
                        "inputs": ("list", {0: {"name": None}}),
                    },
                    "orchestrated": None,
                }
            },
        ),
    }
    response = client.evaluations.update_setup(
        id="ev_567yza",
        dataset=EvaluationsDatasetRequest(version_id="dsv_6L78pqrdFi2xa"),
        evaluatees=[EvaluateeRequest(version_id="prv_7ZlQREDScH0xkhUwtXruN", orchestrated=False)],
        evaluators=[EvaluationsRequest(version_id="evv_012def", orchestrated=False)],
    )
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.evaluations.update_setup(
        id="ev_567yza",
        dataset=EvaluationsDatasetRequest(version_id="dsv_6L78pqrdFi2xa"),
        evaluatees=[EvaluateeRequest(version_id="prv_7ZlQREDScH0xkhUwtXruN", orchestrated=False)],
        evaluators=[EvaluationsRequest(version_id="evv_012def", orchestrated=False)],
    )
    validate_response(async_response, expected_response, expected_types)


async def test_update_status(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "id": "id",
        "dataset": {
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
            "datapoints": [{"id": "id"}],
        },
        "evaluatees": [
            {
                "version": {
                    "path": "path",
                    "id": "id",
                    "name": "name",
                    "version_id": "version_id",
                    "created_at": "2024-01-15T09:30:00Z",
                    "updated_at": "2024-01-15T09:30:00Z",
                    "status": "uncommitted",
                    "last_used_at": "2024-01-15T09:30:00Z",
                    "model": "model",
                    "version_logs_count": 1,
                    "total_logs_count": 1,
                    "inputs": [{"name": "name"}],
                },
                "batch_id": "batch_id",
                "orchestrated": True,
            }
        ],
        "evaluators": [
            {
                "version": {
                    "path": "path",
                    "id": "id",
                    "name": "name",
                    "version_id": "version_id",
                    "created_at": "2024-01-15T09:30:00Z",
                    "updated_at": "2024-01-15T09:30:00Z",
                    "status": "uncommitted",
                    "last_used_at": "2024-01-15T09:30:00Z",
                    "spec": {"arguments_type": "target_free", "return_type": "boolean"},
                    "version_logs_count": 1,
                    "total_logs_count": 1,
                    "inputs": [{"name": "name"}],
                },
                "orchestrated": True,
            }
        ],
        "status": "pending",
        "created_at": "2024-01-15T09:30:00Z",
        "created_by": {"id": "id", "email_address": "email_address", "full_name": "full_name"},
        "updated_at": "2024-01-15T09:30:00Z",
    }
    expected_types: typing.Any = {
        "id": None,
        "dataset": {
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
            "datapoints": ("list", {0: {"id": None}}),
        },
        "evaluatees": (
            "list",
            {
                0: {
                    "version": {
                        "path": None,
                        "id": None,
                        "name": None,
                        "version_id": None,
                        "created_at": "datetime",
                        "updated_at": "datetime",
                        "status": None,
                        "last_used_at": "datetime",
                        "model": None,
                        "version_logs_count": "integer",
                        "total_logs_count": "integer",
                        "inputs": ("list", {0: {"name": None}}),
                    },
                    "batch_id": None,
                    "orchestrated": None,
                }
            },
        ),
        "evaluators": (
            "list",
            {
                0: {
                    "version": {
                        "path": None,
                        "id": None,
                        "name": None,
                        "version_id": None,
                        "created_at": "datetime",
                        "updated_at": "datetime",
                        "status": None,
                        "last_used_at": "datetime",
                        "spec": {"arguments_type": None, "return_type": None},
                        "version_logs_count": "integer",
                        "total_logs_count": "integer",
                        "inputs": ("list", {0: {"name": None}}),
                    },
                    "orchestrated": None,
                }
            },
        ),
        "status": None,
        "created_at": "datetime",
        "created_by": {"id": None, "email_address": None, "full_name": None},
        "updated_at": "datetime",
    }
    response = client.evaluations.update_status(id="id", status="pending")
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.evaluations.update_status(id="id", status="pending")
    validate_response(async_response, expected_response, expected_types)


async def test_get_stats(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "overall_stats": {"num_datapoints": 1, "total_logs": 1, "total_evaluator_logs": 1},
        "version_stats": [
            {
                "version_id": "version_id",
                "num_logs": 1,
                "evaluator_version_stats": [
                    {
                        "evaluator_version_id": "evaluator_version_id",
                        "total_logs": 1,
                        "num_judgments": 1,
                        "num_nulls": 1,
                        "num_errors": 1,
                        "mean": 0,
                        "std": 1,
                        "percentiles": {"0": -2.5, "25": -0.6745, "50": 0, "75": 0.6745, "100": 2.5},
                    }
                ],
            }
        ],
    }
    expected_types: typing.Any = {
        "overall_stats": {"num_datapoints": "integer", "total_logs": "integer", "total_evaluator_logs": "integer"},
        "version_stats": (
            "list",
            {
                0: {
                    "version_id": None,
                    "num_logs": "integer",
                    "evaluator_version_stats": (
                        "list",
                        {
                            0: {
                                "evaluator_version_id": None,
                                "total_logs": "integer",
                                "num_judgments": "integer",
                                "num_nulls": "integer",
                                "num_errors": "integer",
                                "mean": None,
                                "std": None,
                                "percentiles": (
                                    "dict",
                                    {
                                        0: (None, None),
                                        1: (None, None),
                                        2: (None, None),
                                        3: (None, None),
                                        4: (None, None),
                                    },
                                ),
                            }
                        },
                    ),
                }
            },
        ),
    }
    response = client.evaluations.get_stats(id="id")
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.evaluations.get_stats(id="id")
    validate_response(async_response, expected_response, expected_types)


async def test_get_logs(client: Humanloop, async_client: AsyncHumanloop) -> None:
    expected_response: typing.Any = {
        "records": [
            {
                "evaluated_version": {
                    "path": "path",
                    "id": "id",
                    "name": "name",
                    "version_id": "version_id",
                    "created_at": "2024-01-15T09:30:00Z",
                    "updated_at": "2024-01-15T09:30:00Z",
                    "status": "uncommitted",
                    "last_used_at": "2024-01-15T09:30:00Z",
                    "model": "model",
                    "version_logs_count": 1,
                    "total_logs_count": 1,
                    "inputs": [{"name": "name"}],
                },
                "datapoint": {"id": "id"},
                "log": {
                    "id": "id",
                    "prompt": {
                        "path": "path",
                        "id": "id",
                        "name": "name",
                        "version_id": "version_id",
                        "created_at": "2024-01-15T09:30:00Z",
                        "updated_at": "2024-01-15T09:30:00Z",
                        "status": "uncommitted",
                        "last_used_at": "2024-01-15T09:30:00Z",
                        "model": "model",
                        "version_logs_count": 1,
                        "total_logs_count": 1,
                        "inputs": [{"name": "name"}],
                    },
                },
                "evaluator_logs": [
                    {
                        "id": "id",
                        "prompt": {
                            "path": "path",
                            "id": "id",
                            "name": "name",
                            "version_id": "version_id",
                            "created_at": "2024-01-15T09:30:00Z",
                            "updated_at": "2024-01-15T09:30:00Z",
                            "status": "uncommitted",
                            "last_used_at": "2024-01-15T09:30:00Z",
                            "model": "model",
                            "version_logs_count": 1,
                            "total_logs_count": 1,
                            "inputs": [{"name": "name"}],
                        },
                    }
                ],
            }
        ],
        "page": 1,
        "size": 1,
        "total": 1,
    }
    expected_types: typing.Any = {
        "records": (
            "list",
            {
                0: {
                    "evaluated_version": {
                        "path": None,
                        "id": None,
                        "name": None,
                        "version_id": None,
                        "created_at": "datetime",
                        "updated_at": "datetime",
                        "status": None,
                        "last_used_at": "datetime",
                        "model": None,
                        "version_logs_count": "integer",
                        "total_logs_count": "integer",
                        "inputs": ("list", {0: {"name": None}}),
                    },
                    "datapoint": {"id": None},
                    "log": {
                        "id": None,
                        "prompt": {
                            "path": None,
                            "id": None,
                            "name": None,
                            "version_id": None,
                            "created_at": "datetime",
                            "updated_at": "datetime",
                            "status": None,
                            "last_used_at": "datetime",
                            "model": None,
                            "version_logs_count": "integer",
                            "total_logs_count": "integer",
                            "inputs": ("list", {0: {"name": None}}),
                        },
                    },
                    "evaluator_logs": (
                        "list",
                        {
                            0: {
                                "id": None,
                                "prompt": {
                                    "path": None,
                                    "id": None,
                                    "name": None,
                                    "version_id": None,
                                    "created_at": "datetime",
                                    "updated_at": "datetime",
                                    "status": None,
                                    "last_used_at": "datetime",
                                    "model": None,
                                    "version_logs_count": "integer",
                                    "total_logs_count": "integer",
                                    "inputs": ("list", {0: {"name": None}}),
                                },
                            }
                        },
                    ),
                }
            },
        ),
        "page": "integer",
        "size": "integer",
        "total": "integer",
    }
    response = client.evaluations.get_logs(id="id")
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.evaluations.get_logs(id="id")
    validate_response(async_response, expected_response, expected_types)
