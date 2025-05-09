from humanloop.client import Humanloop
from tests.integration.conftest import TestIdentifiers


def test_prompts_call(
    humanloop_test_client: Humanloop,
    prompt: TestIdentifiers,
    test_prompt_config: TestIdentifiers,
) -> None:
    response = humanloop_test_client.prompts.call(
        path=prompt.file_path,
        prompt={**test_prompt_config},
        inputs={"question": "What is the capital of the France?"},
    )
    assert response is not None
    assert response.log_id is not None
    assert response.logs is not None
    for log in response.logs:
        assert log is not None
        assert log.output or log.error or log.output_message is not None
        assert "Paris" in log.output
    assert response.prompt.path == prompt.file_path


def test_prompts_call_stream(
    humanloop_test_client: Humanloop,
    prompt: TestIdentifiers,
    test_prompt_config: TestIdentifiers,
) -> None:
    response = humanloop_test_client.prompts.call_stream(
        path=prompt.file_path,
        prompt={**test_prompt_config},
        inputs={"question": "What is the capital of the France?"},
    )

    output = ""
    for chunk in response:
        assert chunk is not None
        assert chunk.output or chunk.error or chunk.output_message is not None
        assert chunk.id is not None
        assert chunk.prompt_id is not None
        assert chunk.version_id is not None
        output += chunk.output

    assert "Paris" in output
