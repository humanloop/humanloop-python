from humanloop.client import Humanloop

from tests.integration.conftest import TestIdentifiers


def test_prompts_call(
    humanloop_test_client: Humanloop,
    prompt: TestIdentifiers,
    test_prompt_config: TestIdentifiers,
) -> None:
    response = humanloop_test_client.prompts.call(  # type: ignore [attr-defined]
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
    humanloop_test_client: Humanloop,
    prompt: TestIdentifiers,
    test_prompt_config: TestIdentifiers,
) -> None:
    response = humanloop_test_client.prompts.call_stream(  # type: ignore [attr-defined]
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


def test_prompts_call_with_different_questions(
    humanloop_test_client: Humanloop,
    prompt: TestIdentifiers,
    test_prompt_config: TestIdentifiers,
) -> None:
    # Test with a math question
    response = humanloop_test_client.prompts.call(  # type: ignore [attr-defined]
        path=prompt.file_path,
        prompt={**test_prompt_config},  # type: ignore [misc, arg-type, typeddict-item, dict-item, list-item]
        inputs={"question": "What is 5 + 7?"},
    )
    assert response is not None
    assert response.logs is not None
    for log in response.logs:
        assert log is not None
        assert log.output is not None
        assert "12" in log.output

    # Test with a different geography question
    response = humanloop_test_client.prompts.call(  # type: ignore [attr-defined]
        path=prompt.file_path,
        prompt={**test_prompt_config},  # type: ignore [misc, arg-type, typeddict-item, dict-item, list-item]
        inputs={"question": "What is the capital of Japan?"},
    )
    assert response is not None
    assert response.logs is not None
    for log in response.logs:
        assert log is not None
        assert log.output is not None
        assert "Tokyo" in log.output


def test_prompts_call_with_modified_config(
    humanloop_test_client: Humanloop,
    prompt: TestIdentifiers,
    test_prompt_config: TestIdentifiers,
) -> None:
    # Test with modified temperature
    modified_config = {**test_prompt_config}
    modified_config["temperature"] = 0.1  # Lower temperature
    
    response = humanloop_test_client.prompts.call(  # type: ignore [attr-defined]
        path=prompt.file_path,
        prompt=modified_config,  # type: ignore [misc, arg-type, typeddict-item, dict-item, list-item]
        inputs={"question": "What is the capital of the France?"},
    )
    assert response is not None
    assert response.logs is not None
    for log in response.logs:
        assert log is not None
        assert log.output is not None
        assert "Paris" in log.output
    
    # Verify the modified parameter was applied
    for log in response.logs:
        if hasattr(log, "provider_request") and log.provider_request:
            if hasattr(log.provider_request, "temperature"):
                assert log.provider_request.temperature == 0.1


def test_prompts_call_with_tool_choice(
    humanloop_test_client: Humanloop,
    prompt: TestIdentifiers,
    test_prompt_config: TestIdentifiers,
) -> None:
    # Define a simple tool
    calculator_tool = {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to calculate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
    
    # Test with required tool use
    response = humanloop_test_client.prompts.call(  # type: ignore [attr-defined]
        path=prompt.file_path,
        prompt={**test_prompt_config},  # type: ignore [misc, arg-type, typeddict-item, dict-item, list-item]
        inputs={"question": "What is 25 × 4?"},
        tools=[calculator_tool],
        tool_choice={"type": "function", "function": {"name": "calculator"}}
    )
    
    assert response is not None
    assert response.logs is not None
    for log in response.logs:
        assert log is not None
        if hasattr(log, "tool_calls") and log.tool_calls:
            assert len(log.tool_calls) > 0
            tool_call = log.tool_calls[0]
            assert tool_call.function.name == "calculator"
            # Check that the function parameters contain an expression
            assert "expression" in tool_call.function.arguments
