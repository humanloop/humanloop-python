import os
import typing
from typing import Any, List, Optional, Sequence

import httpx
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer

from humanloop.types.model_endpoints import ModelEndpoints
from humanloop.types.model_providers import ModelProviders
from humanloop.types.prompt_kernel_request_stop import PromptKernelRequestStop
from humanloop.types.prompt_kernel_request_template import PromptKernelRequestTemplate
from humanloop.types.response_format import ResponseFormat

from .base_client import AsyncBaseHumanloop, BaseHumanloop
from .decorators.flow import flow as flow_decorator_factory
from .decorators.prompt import prompt as prompt_decorator_factory
from .decorators.tool import tool as tool_decorator_factory
from .environment import HumanloopEnvironment
from .eval_utils import Dataset, Evaluator, EvaluatorCheck, File, _run_eval
from .evaluations.client import EvaluationsClient
from .otel import instrument_provider, set_humanloop_sdk_tracer
from .otel.exporter import HumanloopSpanExporter
from .otel.processor import HumanloopSpanProcessor
from .prompt_utils import populate_template
from .prompts.client import PromptsClient


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
    populate_template = staticmethod(populate_template)


class Humanloop(BaseHumanloop):
    """
    See docstring of :class:`BaseHumanloop`.

    This class extends the base client with custom evaluation utilities
    and decorators for declaring Files in code.
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
        opentelemetry_tracer_provider: Optional[TracerProvider] = None,
        opentelemetry_tracer: Optional[Tracer] = None,
    ):
        """See docstring of :func:`BaseHumanloop.__init__(...)`

        This class extends the base client with custom evaluation utilities
        and decorators for declaring Files in code.

        The Humanloop SDK File decorators use OpenTelemetry internally. You can provide a
        TracerProvider and a Tracer if you'd like to integrate them with your existing
        telemetry system. Otherwise, an internal TracerProvider will be used.
        If you provide only the `TraceProvider`, the SDK will log under a Tracer
        named `humanloop.sdk`.
        """
        super().__init__(
            base_url=base_url,
            environment=environment,
            api_key=api_key,
            timeout=timeout,
            follow_redirects=follow_redirects,
            httpx_client=httpx_client,
        )

        if opentelemetry_tracer_provider is not None:
            self._tracer_provider = opentelemetry_tracer_provider
        else:
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

        if opentelemetry_tracer is not None:
            set_humanloop_sdk_tracer(opentelemetry_tracer)
        else:
            tracer = self._tracer_provider.get_tracer("humanloop.sdk")
            set_humanloop_sdk_tracer(tracer)

        eval_client = ExtendedEvalsClient(client_wrapper=self._client_wrapper)
        eval_client.client = self
        self.evaluations = eval_client
        self.prompts = ExtendedPromptsClient(client_wrapper=self._client_wrapper)

    def prompt(
        self,
        *,
        path: Optional[str] = None,
        model: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
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
        """Decorator for declaring a (Prompt)[https://humanloop.com/docs/explanation/prompts] in code.

        The decorator intercepts calls to LLM provider APIs and creates
        a new Prompt file based on the hyperparameters used in the call.
        If a hyperparameter is specified in the `@prompt` decorator, then
        they override any value intercepted from the LLM provider call.

        If the (Prompt)[https://humanloop.com/docs/explanation/prompts] already exists
        on the specified path, a new version will be upserted when any of the above change.

        Here's an example of declaring a (Prompt)[https://humanloop.com/docs/explanation/prompts] in code:

        ```python
        @prompt(template="You are an assistant on the following topics: {topics}.")
        def call_llm(messages):
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            return client.chat.completions.create(
                model="gpt-4o",
                temperature=0.8,
                frequency_penalty=0.5,
                max_tokens=200,
                messages=messages,
            ).choices[0].message.content
        ```

        This will create a (Prompt)[https://humanloop.com/docs/explanation/prompts] with the following attributes:

        ```python
        {
            model: "gpt-4o",
            endpoint: "chat",
            template: "You are an assistant on the following topics: {topics}.",
            provider: "openai",
            max_tokens: 200,
            temperature: 0.8,
            frequency_penalty: 0.5,
        }

        Every call to the decorated function will create a Log against the Prompt. For example:

        ```python
        call_llm(messages=[
            {"role": "system", "content": "You are an assistant on the following topics: finance."}
            {"role": "user", "content": "What can you do?"}
        ])
        ```

        The Prompt Log will be created with the following inputs:
        ```python
        {
            "inputs": {
                "topics": "finance"
            },
            messages: [
                {"role": "system", "content": "You are an assistant on the following topics: finance."}
                {"role": "user", "content": "What can you do?"}
            ]
            "output": "Hello, I'm an assistant that can help you with anything related to finance."
        }
        ```

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
        return prompt_decorator_factory(
            path=path,
            model=model,
            attributes=attributes,
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
        """Decorator for declaring a [Tool](https://humanloop.com/docs/explanation/tools) in code.

        The decorator inspects the wrapped function's source code, name,
        argument type hints and docstring to infer the values that define
        the [Tool](https://humanloop.com/docs/explanation/tools).

        If the [Tool](https://humanloop.com/docs/explanation/tools) already exists
        on the specified path, a new version will be upserted when any of the
        above change.

        Here's an example of declaring a [Tool](https://humanloop.com/docs/explanation/tools) in code:

        ```python
        @tool
        def calculator(a: int, b: Optional[int]) -> int:
            \"\"\"Add two numbers together.\"\"\"
            return a + b
        ```

        This will create a [Tool](https://humanloop.com/docs/explanation/tools) with the following attributes:
        ```python
        {
            strict: True,
            function: {
                "name": "calculator",
                "description": "Add two numbers together.",
                "parameters": {
                    type: "object",
                    properties: {
                        a: {type: "integer"},
                        b: {type: "integer"}
                    },
                    required: ["a"],
                },
            }
        }
        ```

        Every call to the decorated function will create a Log against the Tool. For example:

        ```python
        calculator(a=1, b=2)
        ```

        Will create the following Log:

        ```python
        {
            "inputs": {
                a: 1,
                b: 2
            },
            "output": 3
        }

        :param path: The path to the Tool. If not provided, the function name
            will be used as the path and the File will be created in the root
            of your Humanloop's organization workspace.

        :param setup_values: Values needed to setup the Tool, defined in
            JSON Schema format: https://json-schema.org/

        :param attributes: Additional fields to describe the Tool.
            Helpful to separate Tool versions from each other
            with details on how they were created or used.
        """
        return tool_decorator_factory(
            path=path,
            setup_values=setup_values,
            attributes=attributes,
        )

    def flow(
        self,
        *,
        path: Optional[str] = None,
        attributes: Optional[dict[str, typing.Any]] = None,
    ):
        """Decorator for declaring a [Flow](https://humanloop.com/docs/explanation/flows) in code.

        A [Flow](https://humanloop.com/docs/explanation/flows) decorator should be added
        at the entrypoint of your LLM feature. Call other functions decorated with
        Humanloop SDK decorators to create a Trace of Logs on Humanloop.

        Here's an example of declaring a [Flow](https://humanloop.com/docs/explanation/flows) in code:
        ```python
        @prompt(template="You are an assistant on the following topics: {topics}.")
        def call_llm(messages):
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            return client.chat.completions.create(
                model="gpt-4o",
                temperature=0.8,
                frequency_penalty=0.5,
                max_tokens=200,
                messages=messages,
            ).choices[0].message.content

        @flow(attributes={"version": "v1"})
        def entrypoint():
            while True:
                messages = []
                user_input = input("You: ")
                if user_input == "exit":
                    break
                messages.append({"role": "user", "content": user_input})
                response = call_llm(messages)
                messages.append({"role": "assistant", "content": response})
                print(f"Assistant: {response}")
        ```

        In this example, the Flow instruments a conversational agent where the
        Prompt defined in `call_llm` is called multiple times in a loop. Calling
        `entrypoint` will create a Flow Trace under which multiple Prompt Logs
        will be nested, allowing you to track the whole conversation session
        between the user and the assistant.

        :param path: The path to the Flow. If not provided, the function name
            will be used as the path and the File will be created in the root
            of your Humanloop's organization workspace.

        :param attributes: A key-value object identifying the Flow Version.
        """
        if attributes is None:
            attributes = {}
        return flow_decorator_factory(path=path, attributes=attributes)


class AsyncHumanloop(AsyncBaseHumanloop):
    """
    See docstring of AsyncBaseHumanloop.

    TODO: Add custom evaluation utilities for async case.
    """

    pass
