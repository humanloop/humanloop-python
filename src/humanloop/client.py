import typing
import os
import httpx
from functools import partial

from .base_client import BaseHumanloop, AsyncBaseHumanloop
from .environment import HumanloopEnvironment
from .eval_utils import _run_eval


class Humanloop(BaseHumanloop):
    """
    See docstring of BaseHumanloop.

    This class extends the base client that contains the auto generated SDK functionality with custom evaluation utilities.
    """

    def __init__(
        self,
        *,
        base_url: typing.Optional[str] = None,
        environment: HumanloopEnvironment = HumanloopEnvironment.DEFAULT,
        api_key: typing.Optional[str] = os.getenv("HUMANLOOP_API_KEY"),
        timeout: typing.Optional[float] = None,
        follow_redirects: typing.Optional[bool] = True,
        httpx_client: typing.Optional[httpx.Client] = None,
    ):
        """See docstring of BaseHumanloop.__init__(...)

        This method extends the base client with evaluation utilities.
        """
        super().__init__(
            base_url=base_url,
            environment=environment,
            api_key=api_key,
            timeout=timeout,
            follow_redirects=follow_redirects,
            httpx_client=httpx_client,
        )
        self.evaluations.run_local = partial(_run_eval, client=self)  # type: ignore[attr-defined]


class AsyncHumanloop(AsyncBaseHumanloop):
    """
    See docstring of AsyncBaseHumanloop.

    TODO: Add custom evaluation utilities for async case.
    """

    pass