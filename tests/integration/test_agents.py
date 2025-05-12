from humanloop.client import Humanloop
from humanloop.types.chat_message import ChatMessage

from tests.integration.conftest import TestIdentifiers


def test_agents_call(
    humanloop_test_client: Humanloop,
    prompt: TestIdentifiers,
    test_prompt_config: TestIdentifiers,
) -> None:
    # Create a test agent configuration
    agent_config = {
        "provider": test_prompt_config['provider'],
        "model": test_prompt_config['model'], 
        "temperature": test_prompt_config['temperature'],
        "template": test_prompt_config['template'],
        "max_iterations": 1,
    }
    
    # Define a simple agent path for testing
    agent_path = "test_agent"
    
    # Test with a simple question
    response = humanloop_test_client.agents.call(  # type: ignore [attr-defined]
        path=agent_path,
        agent=agent_config,
        inputs={"question": "What is the capital of France?"},
    )
    
    assert response is not None
    assert response.log_id is not None
    assert response.output_message is not None
    assert "Paris" in response.output_message.content


def test_agents_call_stream(
    humanloop_test_client: Humanloop,
    prompt: TestIdentifiers,
    test_prompt_config: TestIdentifiers,
) -> None:
    # Create a test agent configuration
    agent_config = {
        "provider": test_prompt_config['provider'],
        "model": test_prompt_config['model'], 
        "temperature": test_prompt_config['temperature'],
        "template": test_prompt_config['template'],
        "max_iterations": 1,
    }
    
    # Define a simple agent path for testing
    agent_path = "test_agent_stream"
    
    # Test with streaming response
    response = humanloop_test_client.agents.call_stream(  # type: ignore [attr-defined]
        path=agent_path,
        agent=agent_config,
        inputs={"question": "What is the capital of France?"},
    )
    
    output = ""
    log_id = None
    
    for chunk in response:
        assert chunk is not None
        if not log_id and chunk.log_id:
            log_id = chunk.log_id
        
        if chunk.payload and chunk.payload.output:
            output += chunk.payload.output
    
    assert log_id is not None  
    assert "Paris" in output


def test_agents_call_with_different_questions(
    humanloop_test_client: Humanloop,
    prompt: TestIdentifiers,
    test_prompt_config: TestIdentifiers,
) -> None:
    # Create a test agent configuration
    agent_config = {
        "provider": test_prompt_config['provider'],
        "model": test_prompt_config['model'], 
        "temperature": test_prompt_config['temperature'],
        "template": test_prompt_config['template'],
        "max_iterations": 1,
    }
    
    agent_path = "test_agent_questions"
    
    # Test with a math question
    response = humanloop_test_client.agents.call(  # type: ignore [attr-defined]
        path=agent_path,
        agent=agent_config,
        inputs={"question": "What is 5 + 7?"},
    )
    
    assert response is not None
    assert response.output_message is not None
    assert "12" in response.output_message.content
    
    # Test with a different geography question
    response = humanloop_test_client.agents.call(  # type: ignore [attr-defined]
        path=agent_path,
        agent=agent_config,
        inputs={"question": "What is the capital of Japan?"},
    )
    
    assert response is not None
    assert response.output_message is not None
    assert "Tokyo" in response.output_message.content


def test_agents_call_with_modified_config(
    humanloop_test_client: Humanloop,
    prompt: TestIdentifiers,
    test_prompt_config: TestIdentifiers,
) -> None:
    # Create a test agent configuration with modified temperature
    agent_config = {
        "provider": test_prompt_config['provider'],
        "model": test_prompt_config['model'], 
        "temperature": 0.1,  # Lower temperature
        "template": test_prompt_config['template'],
        "max_iterations": 1,
    }
    
    agent_path = "test_agent_modified"
    
    response = humanloop_test_client.agents.call(  # type: ignore [attr-defined]
        path=agent_path,
        agent=agent_config,
        inputs={"question": "What is the capital of France?"},
    )
    
    assert response is not None
    assert response.output_message is not None
    assert "Paris" in response.output_message.content


def test_agents_log(
    humanloop_test_client: Humanloop,
    prompt: TestIdentifiers,
    test_prompt_config: TestIdentifiers,
) -> None:
    # Create a test agent configuration
    agent_config = {
        "provider": test_prompt_config['provider'],
        "model": test_prompt_config['model'], 
        "temperature": test_prompt_config['temperature'],
        "template": test_prompt_config['template'],
        "max_iterations": 1,
    }
    
    agent_path = "test_agent_log"
    
    # Test logging agent interaction
    response = humanloop_test_client.agents.log(  # type: ignore [attr-defined]
        path=agent_path,
        agent=agent_config,
        messages=[{"role": "user", "content": "Hello"}],
        output="Hello! How can I assist you today?",
    )
    
    assert response is not None
    assert response.id is not None
    # For CreateAgentLogResponse, the id is used instead of log_id


def test_agents_upsert(
    humanloop_test_client: Humanloop,
    sdk_test_dir: str,
) -> None:
    agent_path = f"{sdk_test_dir}/test_agent_upsert"
    
    # Create a test agent config for upsert
    agent_config = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "template": [
            {
                "role": "system",
                "content": "You are a helpful assistant specialized in answering geography questions.",
            }
        ],
        "max_iterations": 3,
    }
    
    # Test upserting an agent
    response = humanloop_test_client.agents.upsert(  # type: ignore [attr-defined]
        path=agent_path,
        **agent_config,
    )
    
    assert response is not None
    assert response.id is not None
    assert response.path == agent_path
    
    # Clean up after test
    humanloop_test_client.agents.delete(id=response.id)


def test_agents_call_with_tool(
    humanloop_test_client: Humanloop,
    prompt: TestIdentifiers,
    test_prompt_config: TestIdentifiers,
) -> None:
    # Define a simple tool in the inline format
    calculator_tool = {
        "type": "inline",
        "json_schema": {
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
    
    # Create a test agent configuration with tools included
    agent_config = {
        "provider": test_prompt_config['provider'],
        "model": test_prompt_config['model'], 
        "temperature": test_prompt_config['temperature'],
        "template": test_prompt_config['template'],
        "max_iterations": 1,
        "tools": [calculator_tool],  # Include tools in agent config
    }
    
    agent_path = "test_agent_with_tool"
    
    # Test with tool
    response = humanloop_test_client.agents.call(  # type: ignore [attr-defined]
        path=agent_path,
        agent=agent_config,
        inputs={"question": "Calculate 25 × 4"},
        tool_choice={"type": "function", "function": {"name": "calculator"}}
    )
    
    assert response is not None
    assert response.output_message is not None
    # The model should have made a tool call
    assert hasattr(response.output_message, 'tool_calls') and response.output_message.tool_calls
    for tool_call in response.output_message.tool_calls:
        if tool_call.function.name == "calculator":
            assert "expression" in tool_call.function.arguments


def test_agents_continue_call(
    humanloop_test_client: Humanloop,
    prompt: TestIdentifiers,
    test_prompt_config: TestIdentifiers,
) -> None:
    # Define a simple tool in the inline format
    calculator_tool = {
        "type": "inline",
        "json_schema": {
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
    
    # Create a test agent configuration with tools included
    agent_config = {
        "provider": test_prompt_config['provider'],
        "model": test_prompt_config['model'], 
        "temperature": test_prompt_config['temperature'],
        "template": test_prompt_config['template'],
        "max_iterations": 2,  # Need at least 2 for continue call
        "tools": [calculator_tool],  # Include tools in agent config
    }
    
    agent_path = "test_agent_continue"
    
    # First make an initial call that will use a tool
    initial_response = humanloop_test_client.agents.call(  # type: ignore [attr-defined]
        path=agent_path,
        agent=agent_config,
        inputs={"question": "Calculate 25 × 4"},
        tool_choice={"type": "function", "function": {"name": "calculator"}}
    )
    
    assert initial_response is not None
    assert initial_response.log_id is not None
    
    # Get the tool call from output_message
    assert hasattr(initial_response.output_message, 'tool_calls') and initial_response.output_message.tool_calls
    tool_call = None
    for tc in initial_response.output_message.tool_calls:
        if tc.function.name == "calculator":
            tool_call = tc
            break
    
    assert tool_call is not None
    assert tool_call.id is not None
    
    # Now continue the call with a tool response
    continue_response = humanloop_test_client.agents.continue_call(  # type: ignore [attr-defined]
        log_id=initial_response.log_id,
        messages=[{
            "role": "tool", 
            "content": '{"result": 100}',
            "tool_call_id": tool_call.id
        }],
    )
    
    assert continue_response is not None
    assert continue_response.log_id == initial_response.log_id
    assert continue_response.output_message is not None
    assert "100" in continue_response.output_message.content