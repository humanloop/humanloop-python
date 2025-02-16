# """This script demonstrates instrumenting a simple conversational agent with function calling.

# The example uses the Humanloop SDK to declare Files in code.

# Type 'exit' to end the conversation.
# """

import time
from typing import ContextManager, TextIO
from unittest.mock import MagicMock, patch
from humanloop import Humanloop
from tests.integration.chat_agent.conftest import SurferAgentScenario
from tests.integration.conftest import DirectoryIdentifiers


@patch("builtins.input")
def test_scenario_runs(
    mocked_input: MagicMock,
    surfer_agent_scenario: SurferAgentScenario,
    capture_stdout: ContextManager[TextIO],
    humanloop_client: Humanloop,
    test_directory: DirectoryIdentifiers,
):
    scenario_io = [
        "How are you?",
        "Tubular",
        "exit",
    ]
    mocked_input.side_effect = scenario_io
    with capture_stdout() as console_output:  # type: ignore [operator]
        surfer_agent_scenario.agent_chat_workflow()

    time.sleep(5)

    lines = console_output.getvalue().splitlines()
    assert len(lines) == 2
    assert "Agent" in lines[0]

    response = humanloop_client.directories.get(test_directory.id)
    assert len(response.files) == 2
    flow_file = [file for file in response.files if file.type == "flow"][0]
    flow_logs = humanloop_client.logs.list(file_id=flow_file.id)
    assert flow_logs.items and len(flow_logs.items) == 1
    flow_log = flow_logs.items[0]
    assert flow_log.trace_status == "complete"  # type: ignore
    # List will not pass the children to the trace_children attribute
    assert len(flow_log.trace_children) == 0  # type: ignore
    response = humanloop_client.logs.get(flow_log.id)  # type: ignore [assignment]
    if not isinstance(response, dict):
        response = response.dict()  # type: ignore [assignment]
    assert response["trace_status"] == "complete"  # type: ignore
    assert len(response["trace_children"]) == 2  # type: ignore [index]
    messages = response["trace_children"][1]["messages"]  # type: ignore [index]
    assert len(messages) == 4
    # Messages are in reverse order
    assert messages[2]["content"] == scenario_io[0]
    assert messages[0]["content"] == scenario_io[1]
