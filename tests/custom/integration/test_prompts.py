from humanloop.requests.prompt_kernel_request import PromptKernelRequestParams
from tests.custom.integration.conftest import ResourceIdentifiers
from tests.custom.types import GetHumanloopClientFn


def test_prompts_call(
    get_humanloop_client: GetHumanloopClientFn,
    prompt: ResourceIdentifiers,
    test_prompt_config: PromptKernelRequestParams,
) -> None:
    humanloop_client = get_humanloop_client()
    response = humanloop_client.prompts.call(  # type: ignore [attr-defined]
        path=prompt.file_path,
        prompt={**test_prompt_config},  # type: ignore [misc, arg-type, typeddict-item, dict-item, list-item]
        inputs={"question": "What is the capital of the France?"},
    )
    assert response is not None
    assert response.log_id is not None
    assert response.logs is not None
    for log in response.logs:
        assert log is not None
        assert log.output is not None
        assert "Paris" in log.output
    assert response.prompt.path == prompt.file_path


def test_prompts_call_stream(
    get_humanloop_client: GetHumanloopClientFn,
    prompt: ResourceIdentifiers,
    test_prompt_config: PromptKernelRequestParams,
) -> None:
    humanloop_client = get_humanloop_client()
    response = humanloop_client.prompts.call_stream(  # type: ignore [attr-defined]
        path=prompt.file_path,
        prompt={**test_prompt_config},  # type: ignore [misc, arg-type, typeddict-item, dict-item, list-item]
        inputs={"question": "What is the capital of the France?"},
    )

    output = ""
    for chunk in response:
        assert chunk is not None
        assert chunk.output is not None
        assert chunk.id is not None
        assert chunk.prompt_id is not None
        assert chunk.version_id is not None
        output += chunk.output

    assert "Paris" in output
