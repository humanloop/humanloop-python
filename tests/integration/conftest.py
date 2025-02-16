from collections.abc import Callable
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass
from typing import ContextManager
import io
import os
import random
import string
import time
from typing import Generator, TextIO
import pytest
from dotenv import load_dotenv
from humanloop import Humanloop


@dataclass
class APIKeys:
    openai: str
    humanloop: str


@dataclass
class Credentials:
    username: str
    password: str


@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()


@pytest.fixture(scope="session")
def root_integration_directory(humanloop_client: Humanloop) -> Generator[str, None, None]:
    try:
        response = humanloop_client.directories.create(path="SDK_INTEGRATION_TESTS")
    except Exception:
        list_dirs = humanloop_client.directories.list()
        for directory in list_dirs:
            if directory.path == "SDK_INTEGRATION_TESTS":
                _directory_cleanup(directory.id, humanloop_client)
        response = humanloop_client.directories.create(path="SDK_INTEGRATION_TESTS")
    try:
        yield response.path
    finally:
        time.sleep(1)
        _directory_cleanup(response.id, humanloop_client)


@pytest.fixture(scope="session")
def api_keys() -> APIKeys:
    openai_key = os.getenv("OPENAI_API_KEY")
    humanloop_key = os.getenv("HUMANLOOP_API_KEY")
    for key_name, key_value in [
        ("OPENAI_API_KEY", openai_key),
        ("HUMANLOOP_API_KEY", humanloop_key),
    ]:
        if key_value is None:
            raise ValueError(f"{key_name} is not set in .env file")
    api_keys = APIKeys(
        openai=openai_key,
        humanloop=humanloop_key,
    )
    for key, value in asdict(api_keys).items():
        if value is None:
            raise ValueError(f"{key.upper()} key is not set in .env file")
    return api_keys


@pytest.fixture(scope="session")
def humanloop_client(api_keys: APIKeys) -> Humanloop:
    return Humanloop(api_key=api_keys.humanloop)


def _directory_cleanup(directory_id: str, humanloop_client: Humanloop):
    response = humanloop_client.directories.get(directory_id)
    for file in response.files:
        file_id = file.id
        if file.type == "prompt":
            client = humanloop_client.prompts  # type: ignore [assignment]
        elif file.type == "tool":
            client = humanloop_client.tools  # type: ignore [assignment]
        elif file.type == "dataset":
            client = humanloop_client.datasets  # type: ignore [assignment]
        elif file.type == "evaluator":
            client = humanloop_client.evaluators  # type: ignore [assignment]
        elif file.type == "flow":
            client = humanloop_client.flows  # type: ignore [assignment]
        else:
            raise NotImplementedError(f"Unknown HL file type {file.type}")
        client.delete(file_id)

    for subdirectory in response.subdirectories:
        _directory_cleanup(
            directory_id=subdirectory.id,
            humanloop_client=humanloop_client,
        )

    humanloop_client.directories.delete(id=response.id)


@dataclass
class DirectoryIdentifiers:
    path: str
    id: str


@pytest.fixture()
def test_directory(
    root_integration_directory: str,
    humanloop_client: Humanloop,
) -> Generator[DirectoryIdentifiers, None, None]:
    # Generate a random  alphanumeric directory name to avoid conflicts
    def get_random_string(length: int = 32) -> str:
        return "".join([random.choice(string.ascii_letters) for _ in range(length)])

    directory_path = f"{root_integration_directory}/{get_random_string()}"
    response = humanloop_client.directories.create(path=directory_path)
    assert response.path == directory_path
    try:
        yield DirectoryIdentifiers(
            path=response.path,
            id=response.id,
        )
    finally:
        time.sleep(1)
        _directory_cleanup(response.id, humanloop_client)


@pytest.fixture()
def get_test_path(test_directory: DirectoryIdentifiers) -> Callable[[str], str]:
    def generate_path(name: str) -> str:
        return f"{test_directory.path}/{name}"

    return generate_path


@pytest.fixture()
def capture_stdout() -> ContextManager[TextIO]:
    @contextmanager
    def _context_manager():
        f = io.StringIO()
        with redirect_stdout(f):
            yield f

    return _context_manager  # type: ignore [return-value]
