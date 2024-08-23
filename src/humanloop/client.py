# This file was auto-generated by Fern from our API Definition.

import os
import typing

import httpx

from .core.api_error import ApiError
from .core.client_wrapper import AsyncClientWrapper, SyncClientWrapper
from .datasets.client import AsyncDatasetsClient, DatasetsClient
from .environment import HumanloopEnvironment
from .evaluations.client import AsyncEvaluationsClient, EvaluationsClient
from .evaluators.client import AsyncEvaluatorsClient, EvaluatorsClient
from .logs.client import AsyncLogsClient, LogsClient
from .prompts.client import AsyncPromptsClient, PromptsClient
from .sessions.client import AsyncSessionsClient, SessionsClient
from .tools.client import AsyncToolsClient, ToolsClient


class Humanloop:
    """
    Use this class to access the different functions within the SDK. You can instantiate any number of clients with different configuration that will propagate to these functions.

    Parameters
    ----------
    base_url : typing.Optional[str]
        The base url to use for requests from the client.

    environment : HumanloopEnvironment
        The environment to use for requests from the client. from .environment import HumanloopEnvironment



        Defaults to HumanloopEnvironment.DEFAULT



    api_key : typing.Optional[str]
    timeout : typing.Optional[float]
        The timeout to be used, in seconds, for requests. By default the timeout is 60 seconds, unless a custom httpx client is used, in which case this default is not enforced.

    follow_redirects : typing.Optional[bool]
        Whether the default httpx client follows redirects or not, this is irrelevant if a custom httpx client is passed in.

    httpx_client : typing.Optional[httpx.Client]
        The httpx client to use for making requests, a preconfigured client is used by default, however this is useful should you want to pass in any custom httpx configuration.

    Examples
    --------
    from humanloop import Humanloop

    client = Humanloop(
        api_key="YOUR_API_KEY",
    )
    """

    def __init__(
        self,
        *,
        base_url: typing.Optional[str] = None,
        environment: HumanloopEnvironment = HumanloopEnvironment.DEFAULT,
        api_key: typing.Optional[str] = os.getenv("HUMANLOOP_API_KEY"),
        timeout: typing.Optional[float] = None,
        follow_redirects: typing.Optional[bool] = True,
        httpx_client: typing.Optional[httpx.Client] = None
    ):
        _defaulted_timeout = timeout if timeout is not None else 60 if httpx_client is None else None
        if api_key is None:
            raise ApiError(
                body="The client must be instantiated be either passing in api_key or setting HUMANLOOP_API_KEY"
            )
        self._client_wrapper = SyncClientWrapper(
            base_url=_get_base_url(base_url=base_url, environment=environment),
            api_key=api_key,
            httpx_client=httpx_client
            if httpx_client is not None
            else httpx.Client(timeout=_defaulted_timeout, follow_redirects=follow_redirects)
            if follow_redirects is not None
            else httpx.Client(timeout=_defaulted_timeout),
            timeout=_defaulted_timeout,
        )
        self.prompts = PromptsClient(client_wrapper=self._client_wrapper)
        self.tools = ToolsClient(client_wrapper=self._client_wrapper)
        self.datasets = DatasetsClient(client_wrapper=self._client_wrapper)
        self.evaluations = EvaluationsClient(client_wrapper=self._client_wrapper)
        self.evaluators = EvaluatorsClient(client_wrapper=self._client_wrapper)
        self.logs = LogsClient(client_wrapper=self._client_wrapper)
        self.sessions = SessionsClient(client_wrapper=self._client_wrapper)


class AsyncHumanloop:
    """
    Use this class to access the different functions within the SDK. You can instantiate any number of clients with different configuration that will propagate to these functions.

    Parameters
    ----------
    base_url : typing.Optional[str]
        The base url to use for requests from the client.

    environment : HumanloopEnvironment
        The environment to use for requests from the client. from .environment import HumanloopEnvironment



        Defaults to HumanloopEnvironment.DEFAULT



    api_key : typing.Optional[str]
    timeout : typing.Optional[float]
        The timeout to be used, in seconds, for requests. By default the timeout is 60 seconds, unless a custom httpx client is used, in which case this default is not enforced.

    follow_redirects : typing.Optional[bool]
        Whether the default httpx client follows redirects or not, this is irrelevant if a custom httpx client is passed in.

    httpx_client : typing.Optional[httpx.AsyncClient]
        The httpx client to use for making requests, a preconfigured client is used by default, however this is useful should you want to pass in any custom httpx configuration.

    Examples
    --------
    from humanloop import AsyncHumanloop

    client = AsyncHumanloop(
        api_key="YOUR_API_KEY",
    )
    """

    def __init__(
        self,
        *,
        base_url: typing.Optional[str] = None,
        environment: HumanloopEnvironment = HumanloopEnvironment.DEFAULT,
        api_key: typing.Optional[str] = os.getenv("HUMANLOOP_API_KEY"),
        timeout: typing.Optional[float] = None,
        follow_redirects: typing.Optional[bool] = True,
        httpx_client: typing.Optional[httpx.AsyncClient] = None
    ):
        _defaulted_timeout = timeout if timeout is not None else 60 if httpx_client is None else None
        if api_key is None:
            raise ApiError(
                body="The client must be instantiated be either passing in api_key or setting HUMANLOOP_API_KEY"
            )
        self._client_wrapper = AsyncClientWrapper(
            base_url=_get_base_url(base_url=base_url, environment=environment),
            api_key=api_key,
            httpx_client=httpx_client
            if httpx_client is not None
            else httpx.AsyncClient(timeout=_defaulted_timeout, follow_redirects=follow_redirects)
            if follow_redirects is not None
            else httpx.AsyncClient(timeout=_defaulted_timeout),
            timeout=_defaulted_timeout,
        )
        self.prompts = AsyncPromptsClient(client_wrapper=self._client_wrapper)
        self.tools = AsyncToolsClient(client_wrapper=self._client_wrapper)
        self.datasets = AsyncDatasetsClient(client_wrapper=self._client_wrapper)
        self.evaluations = AsyncEvaluationsClient(client_wrapper=self._client_wrapper)
        self.evaluators = AsyncEvaluatorsClient(client_wrapper=self._client_wrapper)
        self.logs = AsyncLogsClient(client_wrapper=self._client_wrapper)
        self.sessions = AsyncSessionsClient(client_wrapper=self._client_wrapper)


def _get_base_url(*, base_url: typing.Optional[str] = None, environment: HumanloopEnvironment) -> str:
    if base_url is not None:
        return base_url
    elif environment is not None:
        return environment.value
    else:
        raise Exception("Please pass in either base_url or environment to construct the client")
