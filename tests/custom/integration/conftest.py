import io
import os
import time
import uuid
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from typing import ContextManager, Generator, List, TextIO, Union

import dotenv
import pytest
from click.testing import CliRunner

from humanloop import AgentResponse, PromptResponse
from humanloop.requests.prompt_kernel_request import PromptKernelRequestParams
from tests.custom.types import GetHumanloopClientFn, SyncableFile


@dataclass
class ResourceIdentifiers:
    file_id: str
    file_path: str


@pytest.fixture()
def capture_stdout() -> ContextManager[TextIO]:
    @contextmanager
    def _context_manager():
        f = io.StringIO()
        with redirect_stdout(f):
            yield f

    return _context_manager  # type: ignore [return-value]


@pytest.fixture(scope="session")
def openai_key() -> str:
    dotenv.load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY is not set for integration tests")
    return os.getenv("OPENAI_API_KEY")  # type: ignore [return-value]


@pytest.fixture(scope="function")
def sdk_test_dir(get_humanloop_client: GetHumanloopClientFn) -> Generator[str, None, None]:
    humanloop_client = get_humanloop_client()

    def cleanup_directory(directory_id: str):
        directory_response = humanloop_client.directories.get(id=directory_id)
        for subdirectory in directory_response.subdirectories:
            cleanup_directory(subdirectory.id)
        for file in directory_response.files:
            match file.type:
                case "agent":
                    humanloop_client.agents.delete(id=file.id)
                case "prompt":
                    humanloop_client.prompts.delete(id=file.id)
                case "dataset":
                    humanloop_client.datasets.delete(id=file.id)
                case "evaluator":
                    humanloop_client.evaluators.delete(id=file.id)
                case "flow":
                    humanloop_client.flows.delete(id=file.id)
                case "tool":
                    humanloop_client.tools.delete(id=file.id)
                case _:
                    raise ValueError(f"Unknown file type: {file.type}")
        humanloop_client.directories.delete(id=directory_response.id)

    path = f"SDK_INTEGRATION_TEST_{uuid.uuid4()}"
    response = None
    try:
        response = humanloop_client.directories.create(path=path)
        yield response.path
    except Exception as e:
        pytest.fail(f"Failed to create directory {path}: {e}")
    finally:
        if response:
            time.sleep(5)
            cleanup_directory(response.id)


@pytest.fixture(scope="function")
def test_prompt_config() -> PromptKernelRequestParams:
    return {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.5,
        "template": [
            {
                "role": "system",
                "content": "You are a helpful assistant. You must answer the user's question truthfully and at the level of a 5th grader.",
            },
            {
                "role": "user",
                "content": "{{question}}",
            },
        ],
    }


@pytest.fixture(scope="function")
def prompt(
    get_humanloop_client: GetHumanloopClientFn,
    sdk_test_dir: str,
    test_prompt_config: PromptKernelRequestParams,
) -> Generator[ResourceIdentifiers, None, None]:
    humanloop_client = get_humanloop_client()
    prompt_path = f"{sdk_test_dir}/prompt"
    try:
        response = humanloop_client.prompts.upsert(
            path=prompt_path,
            **test_prompt_config,
        )
        yield ResourceIdentifiers(file_id=response.id, file_path=response.path)
        humanloop_client.prompts.delete(id=response.id)
    except Exception as e:
        pytest.fail(f"Failed to create prompt {prompt_path}: {e}")


@pytest.fixture(scope="function")
def eval_dataset(
    get_humanloop_client: GetHumanloopClientFn, sdk_test_dir: str
) -> Generator[ResourceIdentifiers, None, None]:
    humanloop_client = get_humanloop_client()
    dataset_path = f"{sdk_test_dir}/eval_dataset"
    try:
        response = humanloop_client.datasets.upsert(
            path=dataset_path,
            datapoints=[
                {
                    "inputs": {
                        "question": "What is the capital of the France?",
                    },
                },
                {
                    "inputs": {
                        "question": "What is the capital of the Germany?",
                    },
                },
                {
                    "inputs": {
                        "question": "What is 2+2?",
                    },
                },
            ],
        )
        yield ResourceIdentifiers(file_id=response.id, file_path=response.path)
        humanloop_client.datasets.delete(id=response.id)
    except Exception as e:
        pytest.fail(f"Failed to create dataset {dataset_path}: {e}")


@pytest.fixture(scope="function")
def eval_prompt(
    get_humanloop_client: GetHumanloopClientFn,
    sdk_test_dir: str,
    openai_key: str,
    test_prompt_config: PromptKernelRequestParams,
) -> Generator[ResourceIdentifiers, None, None]:
    humanloop_client = get_humanloop_client()
    prompt_path = f"{sdk_test_dir}/eval_prompt"
    try:
        response = humanloop_client.prompts.upsert(
            path=prompt_path,
            **test_prompt_config,
        )
        yield ResourceIdentifiers(file_id=response.id, file_path=response.path)
        humanloop_client.prompts.delete(id=response.id)
    except Exception as e:
        pytest.fail(f"Failed to create prompt {prompt_path}: {e}")


@pytest.fixture(scope="function")
def output_not_null_evaluator(
    get_humanloop_client: GetHumanloopClientFn, sdk_test_dir: str
) -> Generator[ResourceIdentifiers, None, None]:
    humanloop_client = get_humanloop_client()
    evaluator_path = f"{sdk_test_dir}/output_not_null_evaluator"
    try:
        response = humanloop_client.evaluators.upsert(
            path=evaluator_path,
            spec={
                "arguments_type": "target_required",
                "return_type": "boolean",
                "code": """
def output_not_null(log: dict) -> bool:
    return log["output"] is not None
                """,
                "evaluator_type": "python",
            },
        )
        yield ResourceIdentifiers(file_id=response.id, file_path=response.path)
        humanloop_client.evaluators.delete(id=response.id)
    except Exception as e:
        pytest.fail(f"Failed to create evaluator {evaluator_path}: {e}")


@pytest.fixture(scope="function")
def id_for_staging_environment(get_humanloop_client: GetHumanloopClientFn, eval_prompt: ResourceIdentifiers) -> str:
    humanloop_client = get_humanloop_client()
    response = humanloop_client.prompts.list_environments(id=eval_prompt.file_id)
    for environment in response:
        if environment.name == "staging":
            return environment.id
    pytest.fail("Staging environment not found")


@pytest.fixture
def syncable_files_fixture(
    get_humanloop_client: GetHumanloopClientFn,
    sdk_test_dir: str,
) -> Generator[list[SyncableFile], None, None]:
    """Creates a predefined structure of files in Humanloop for testing sync."""
    files: List[SyncableFile] = [
        SyncableFile(
            path="prompts/gpt-4",
            type="prompt",
            model="gpt-4",
        ),
        SyncableFile(
            path="prompts/gpt-4o",
            type="prompt",
            model="gpt-4o",
        ),
        SyncableFile(
            path="prompts/nested/complex/gpt-4o",
            type="prompt",
            model="gpt-4o",
        ),
        SyncableFile(
            path="agents/gpt-4",
            type="agent",
            model="gpt-4",
        ),
        SyncableFile(
            path="agents/gpt-4o",
            type="agent",
            model="gpt-4o",
        ),
    ]

    humanloop_client = get_humanloop_client()
    created_files = []
    for file in files:
        full_path = f"{sdk_test_dir}/{file.path}"
        response: Union[AgentResponse, PromptResponse]
        if file.type == "prompt":
            response = humanloop_client.prompts.upsert(
                path=full_path,
                model=file.model,
            )
        elif file.type == "agent":
            response = humanloop_client.agents.upsert(
                path=full_path,
                model=file.model,
            )
        created_files.append(
            SyncableFile(
                path=full_path, type=file.type, model=file.model, id=response.id, version_id=response.version_id
            )
        )

    yield created_files


@pytest.fixture
def cli_runner() -> CliRunner:
    """GIVEN a CLI runner
    THEN it should be configured to catch exceptions
    """
    return CliRunner(mix_stderr=False)


@pytest.fixture
def no_humanloop_api_key_in_env(monkeypatch):
    """Fixture that removes HUMANLOOP_API_KEY from environment variables.

    Use this fixture in tests that verify behavior when no API key is available
    in the environment (but could still be loaded from .env files).
    """
    # Remove API key from environment
    monkeypatch.delenv("HUMANLOOP_API_KEY", raising=False)
    yield
