import typing
from typing import Any, Optional, List, Sequence
import os
import httpx
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from humanloop.types.response_format import ResponseFormat

from .decorators.flow import flow as flow_decorator
from .decorators.prompt import prompt as prompt_decorator
from .decorators.tool import tool as tool_decorator
from humanloop.core.client_wrapper import SyncClientWrapper
from humanloop.types.model_endpoints import ModelEndpoints
from humanloop.types.model_providers import ModelProviders
from humanloop.types.prompt_kernel_request_template import PromptKernelRequestTemplate
from humanloop.types.prompt_kernel_request_stop import PromptKernelRequestStop
from .otel.exporter import HumanloopSpanExporter
from .otel.processor import HumanloopSpanProcessor
from .otel import instrument_provider, set_tracer
from .base_client import BaseHumanloop, AsyncBaseHumanloop
from .environment import HumanloopEnvironment
from .eval_utils import _run_eval, Dataset, File, Evaluator, EvaluatorCheck
from .prompts.client import PromptsClient
from .evaluations.client import EvaluationsClient
from .prompt_utils import populate_template


class ExtendedEvalsClient(EvaluationsClient):
    client: BaseHumanloop

    def run(
        self,
        file: File,
        name: Optional[str],
        dataset: Dataset,
        evaluators: Optional[Sequence[Evaluator]] = None,
        # logs: typing.Sequence[dict] | None = None,
        workers: int = 4,
    ) -> List[EvaluatorCheck]:
        """Evaluate your function for a given `Dataset` and set of `Evaluators`.

        :param file: the Humanloop file being evaluated, including a function to run over the dataset.
        :param name: the name of the Evaluation to run. If it does not exist, a new Evaluation will be created under your File.
        :param dataset: the dataset to map your function over to produce the outputs required by the Evaluation.
        :param evaluators: define how judgments are provided for this Evaluation.
        :param workers: the number of threads to process datapoints using your function concurrently.
        :return: per Evaluator checks.
        """
        if self.client is None:
            raise ValueError("Need Humanloop client defined to run evals")

        return _run_eval(
            client=self.client,
            file=file,
            name=name,
            dataset=dataset,
            evaluators=evaluators,
            workers=workers,
        )


class ExtendedPromptsClient(PromptsClient):
    def __init__(self, client_wrapper: SyncClientWrapper):
        super().__init__(client_wrapper=client_wrapper)

    populate_template = staticmethod(populate_template)


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

        self._tracer_provider = TracerProvider(
            resource=Resource(
                attributes={
                    "instrumentor": "humanloop.sdk",
                }
            ),
        )
        instrument_provider(provider=self._tracer_provider)
        self._tracer_provider.add_span_processor(
            HumanloopSpanProcessor(
                exporter=HumanloopSpanExporter(
                    client=self,
                )
            ),
        )
        tracer = self._tracer_provider.get_tracer("humanloop.sdk")
        set_tracer(tracer)

        eval_client = ExtendedEvalsClient(client_wrapper=self._client_wrapper)
        eval_client.client = self
        self.evaluations = eval_client
        self.prompts = ExtendedPromptsClient(client_wrapper=self._client_wrapper)

    def prompt(
        self,
        *,
        # TODO: Template can be a list of objects
        path: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: Optional[ModelEndpoints] = None,
        template: Optional[PromptKernelRequestTemplate] = None,
        provider: Optional[ModelProviders] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[PromptKernelRequestStop] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        other: Optional[dict[str, Optional[Any]]] = None,
        seed: Optional[int] = None,
        response_format: Optional[ResponseFormat] = None,
    ):
        """Decorator to mark a function as a Humanloop Prompt.

        The decorator intercepts calls to LLM provider APIs, extracts
        hyperparameters used in the call, and upsert a new Prompt File
        on Humanloop based on them. If a hyperparameter is specified in the
        `@prompt` decorator, then it overrides any inference made from inside
        the function.

        If the Prompt already exists on the specified path, a new version will
        be inserted when the hyperparameters used in making LLM calls change.

        :param path: The path where the Prompt is created. If not
            provided, the function name is used as the path and the File
            is created in the root of your Humanloop's organization workspace.

        :param model: Name of the model used by the Prompt.

        :param endpoint: The model instance used, e.g. `gpt-4`. See
            [supported models](https://humanloop.com/docs/reference/supported-models)

        :param template: The template for the Prompt. This is the text of
            the system message used to set the LLM prompt. The template
            accepts template slots using the format `{slot_name}`.

        :param provider: The company providing the underlying model service.

        :param max_tokens: Maximum number of tokens used in generation.

        :param temperature: What sampling temperature to use
            when making a generation. Higher values means the model
            will be more creative.

        :param top_p: An alternative to sampling with temperature,
            called nucleus sampling, where the model considers the results
            of the tokens with top_p probability mass.

        :param stop: Token or list of tokens that stop generation

        :param presence_penalty: Number between -2.0 and 2.0.
            Positive values penalize new tokens based on whether they
            appear in the generation so far.

        :param frequency_penalty: Number between -2.0 and 2.0. Positive
            values penalize new tokens based on how frequently they
            appear in the generation so far.

        :param other: Other parameter values to be passed to the provider call.

        :param seed: If specified, model will make a best effort to
            sample deterministically, but it is not guaranteed.

        :param response_format: The format of the response.
            Only `{"type": "json_object"}` is currently supported
            for chat.
        """
        return prompt_decorator(
            path=path,
            model=model,
            endpoint=endpoint,
            template=template,
            provider=provider,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            other=other,
            seed=seed,
            response_format=response_format,
        )

    def tool(
        self,
        *,
        path: Optional[str] = None,
        setup_values: Optional[dict[str, Optional[Any]]] = None,
        attributes: Optional[dict[str, Optional[Any]]] = None,
    ):
        """Decorator to mark a function as a Humanloop Tool.

        The decorator inspect the wrapped function signature and code to infer
        the File kernel and JSON schema for the Tool. Any change to the decorated
        function will create a new version of the Tool, provided that the path
        remains the same.

        Every call to the decorated function will create a Log against the Tool.

        :param path: The path to the Tool. If not provided, the function name
            will be used as the path and the File will be created in the root
            of your Humanloop's organization workspace.

        :param setup_values: Values needed to setup the Tool, defined in
            JSON Schema format: https://json-schema.org/

        :param attributes: Additional fields to describe the Tool.
            Helpful to separate Tool versions from each other
            with details on how they were created or used.
        """
        return tool_decorator(
            path=path,
            setup_values=setup_values,
            attributes=attributes,
        )

    def flow(
        self,
        *,
        path: Optional[str] = None,
        attributes: dict[str, typing.Any] = {},
    ):
        """Decorator to log a Flow to the Humanloop API.

        The decorator logs the inputs and outputs of the decorated function to
        create a Log against the Flow in Humanloop.

        The decorator is an entrypoint to the instrumented AI feature. Decorated
        functions called in the context of function decorated with Flow will create
        a Trace in Humanloop.

        :param path: The path to the Flow. If not provided, the function name
            will be used as the path and the File will be created in the root
            of your Humanloop's organization workspace.

        :param attributes: A key-value object identifying the Flow Version.
        """
        return flow_decorator(path=path, attributes=attributes)


class AsyncHumanloop(AsyncBaseHumanloop):
    """
    See docstring of AsyncBaseHumanloop.

    TODO: Add custom evaluation utilities for async case.
    """

    pass
