# This file was auto-generated by Fern from our API Definition.

import typing_extensions
import typing
from .agent_response import AgentResponseParams


class ListAgentsParams(typing_extensions.TypedDict):
    records: typing.Sequence[AgentResponseParams]
    """
    The list of Agents.
    """
