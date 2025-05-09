from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
import os
from typing import Any, ContextManager, Generator
import io
from typing import TextIO
import uuid
import pytest
import dotenv
from humanloop.client import Humanloop


@dataclass
class TestIdentifiers:
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


@pytest.fixture(scope="session")
def humanloop_test_client() -> Humanloop:
    dotenv.load_dotenv()
    if not os.getenv("HUMANLOOP_API_KEY"):
        pytest.fail("HUMANLOOP_API_KEY is not set for integration tests")
    return Humanloop(api_key=os.getenv("HUMANLOOP_API_KEY"))  # type: ignore [return-value]


@pytest.fixture(scope="function")
def sdk_test_dir(humanloop_test_client: Humanloop) -> Generator[str, None, None]:
    path = f"SDK_INTEGRATION_TEST_{uuid.uuid4()}"
    try:
        response = humanloop_test_client.directories.create(path=path)
        yield response.path
        humanloop_test_client.directories.delete(id=response.id)
    except Exception as e:
        pytest.fail(f"Failed to create directory {path}: {e}")


@pytest.fixture(scope="function")
def test_prompt_config() -> dict[str, Any]:
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
def eval_dataset(humanloop_test_client: Humanloop, sdk_test_dir: str) -> Generator[TestIdentifiers, None, None]:
    dataset_path = f"{sdk_test_dir}/eval_dataset"
    try:
        response = humanloop_test_client.datasets.upsert(
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
        yield TestIdentifiers(file_id=response.id, file_path=response.path)
        humanloop_test_client.datasets.delete(id=response.id)
    except Exception as e:
        pytest.fail(f"Failed to create dataset {dataset_path}: {e}")


@pytest.fixture(scope="function")
def eval_prompt(
    humanloop_test_client: Humanloop, sdk_test_dir: str, openai_key: str, test_prompt_config: dict[str, Any]
) -> Generator[TestIdentifiers, None, None]:
    prompt_path = f"{sdk_test_dir}/eval_prompt"
    try:
        response = humanloop_test_client.prompts.upsert(
            path=prompt_path,
            **test_prompt_config,
        )
        yield TestIdentifiers(file_id=response.id, file_path=response.path)
        humanloop_test_client.prompts.delete(id=response.id)
    except Exception as e:
        pytest.fail(f"Failed to create prompt {prompt_path}: {e}")


@pytest.fixture(scope="function")
def prompt(
    humanloop_test_client: Humanloop, sdk_test_dir: str, openai_key: str, test_prompt_config: dict[str, Any]
) -> Generator[TestIdentifiers, None, None]:
    prompt_path = f"{sdk_test_dir}/prompt"
    try:
        response = humanloop_test_client.prompts.upsert(
            path=prompt_path,
            **test_prompt_config,
        )
        yield TestIdentifiers(file_id=response.id, file_path=response.path)
        humanloop_test_client.prompts.delete(id=response.id)
    except Exception as e:
        pytest.fail(f"Failed to create prompt {prompt_path}: {e}")


@pytest.fixture(scope="function")
def output_not_null_evaluator(
    humanloop_test_client: Humanloop, sdk_test_dir: str
) -> Generator[TestIdentifiers, None, None]:
    evaluator_path = f"{sdk_test_dir}/output_not_null_evaluator"
    try:
        response = humanloop_test_client.evaluators.upsert(
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
        yield TestIdentifiers(file_id=response.id, file_path=response.path)
        humanloop_test_client.evaluators.delete(id=response.id)
    except Exception as e:
        pytest.fail(f"Failed to create evaluator {evaluator_path}: {e}")


@pytest.fixture(scope="function")
def id_for_staging_environment(humanloop_test_client: Humanloop, eval_prompt: TestIdentifiers) -> str:
    response = humanloop_test_client.prompts.list_environments(id=eval_prompt.file_id)
    for environment in response:
        if environment.name == "staging":
            return environment.id
    pytest.fail("Staging environment not found")
