import os
import typing
from typing import Any, List, Optional, Sequence

import httpx
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer

from humanloop.core.client_wrapper import SyncClientWrapper

from humanloop.evals import run_eval
from humanloop.evals.types import Dataset, Evaluator, EvaluatorCheck, File

from humanloop.base_client import AsyncBaseHumanloop, BaseHumanloop
from humanloop.overload import overload_call, overload_log
from humanloop.decorators.flow import flow as flow_decorator_factory
from humanloop.decorators.prompt import prompt_decorator_factory
from humanloop.decorators.tool import tool_decorator_factory as tool_decorator_factory
from humanloop.environment import HumanloopEnvironment
from humanloop.evaluations.client import EvaluationsClient
from humanloop.otel import instrument_provider
from humanloop.otel.exporter import HumanloopSpanExporter
from humanloop.otel.processor import HumanloopSpanProcessor


class ExtendedEvalsClient(EvaluationsClient):
    """
    Provides high-level utilities for running Evaluations on the local runtime.
    """

    client: BaseHumanloop

    def __init__(
        self,
        *,
        client_wrapper: SyncClientWrapper,
    ):
        super().__init__(client_wrapper=client_wrapper)

    def run(
        self,
        file: File,
        name: Optional[str],
        dataset: Dataset,
        evaluators: Optional[Sequence[Evaluator]] = None,
        workers: int = 4,
    ) -> List[EvaluatorCheck]:
        """Evaluate your function for a given `Dataset` and set of `Evaluators`.

        :param file: the Humanloop file being evaluated, including a function to run over the dataset.
        :param name: the name of the Evaluation to run. If it does not exist, a new Evaluation will be created under your File.
        :param dataset: the dataset to map your function over to produce the outputs required by the Evaluation.
        :param evaluators: define how judgments are provided for this Evaluation.
        :param workers: Number of concurrent threads for processing datapoints.
        :return: per Evaluator checks.
        """
        if self.client is None:
            raise ValueError("Need Humanloop client defined to run evals")

        return run_eval(
            client=self.client,
            file=file,
            name=name,
            dataset=dataset,
            evaluators=evaluators,
            workers=workers,
        )


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
        """
        Extends the base client with custom evaluation utilities and
        decorators for declaring Files in code.

        The Humanloop SDK File decorators use OpenTelemetry internally.
        You can provide a TracerProvider and a Tracer to integrate
        with your existing telemetry system. If not provided,
        an internal TracerProvider will be used.
        """
        super().__init__(
            base_url=base_url,
            environment=environment,
            api_key=api_key,
            timeout=timeout,
            follow_redirects=follow_redirects,
            httpx_client=httpx_client,
        )

        eval_client = ExtendedEvalsClient(client_wrapper=self._client_wrapper)
        eval_client.client = self
        self.evaluations = eval_client

        # Overload the .log method of the clients to be aware of Evaluation Context
        # and the @flow decorator providing the trace_id
        self.prompts = overload_log(client=self.prompts)
        self.prompts = overload_call(client=self.prompts)
        self.flows = overload_log(client=self.flows)
        self.tools = overload_log(client=self.tools)

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
            HumanloopSpanProcessor(exporter=HumanloopSpanExporter(client=self)),
        )

        if opentelemetry_tracer is None:
            self._opentelemetry_tracer = self._tracer_provider.get_tracer(
                "humanloop.sdk"
            )
        else:
            self._opentelemetry_tracer = opentelemetry_tracer

    def prompt(
        self,
        *,
        path: str,
    ):
        """Auto-instrument LLM providers and create [Prompt](https://humanloop.com/docs/explanation/prompts)
        Logs on Humanloop from them.

        ```python
        @prompt(path="My Prompt")
        def call_llm(messages):
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            return client.chat.completions.create(
                model="gpt-4o",
                temperature=0.8,
                frequency_penalty=0.5,
                max_tokens=200,
                messages=messages,
            ).choices[0].message.content

        Calling the function above creates a new Log on Humanloop
        against this Prompt version:
        {
            provider: "openai",
            model: "gpt-4o",
            endpoint: "chat",
            max_tokens: 200,
            temperature: 0.8,
            frequency_penalty: 0.5,
        }
        ```

        If a different model, endpoint, or hyperparameter is used, a new
        Prompt version is created. For example:
        ```
        @humanloop_client.prompt(path="My Prompt")
        def call_llm(messages):
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.5,
            ).choices[0].message.content

            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            client.messages.create(
                model="claude-3-5-sonnet-20240620",
                temperature=0.5,
            ).content

        Calling this function will create two versions of the same Prompt:
        {
            provider: "openai",
            model: "gpt-4o-mini",
            endpoint: "chat",
            max_tokens: 200,
            temperature: 0.5,
            frequency_penalty: 0.5,
        }

        {
            provider: "anthropic",
            model: "claude-3-5-sonnet-20240620",
            endpoint: "messages",
            temperature: 0.5,
        }

        And one Log will be added to each version of the Prompt.
        ```

        :param path: The path where the Prompt is created. If not
            provided, the function name is used as the path and the File
            is created in the root of your Humanloop organization workspace.

        :param prompt_kernel: Attributes that define the Prompt. See `class:DecoratorPromptKernelRequestParams`
        """
        return prompt_decorator_factory(path=path)

    def tool(
        self,
        *,
        path: str,
        attributes: Optional[dict[str, Any]] = None,
        setup_values: Optional[dict[str, Any]] = None,
    ):
        """Manage [Tool](https://humanloop.com/docs/explanation/tools) Files through code.

        The decorator inspects the wrapped function's source code to infer the Tool's
        JSON Schema. If the function declaration changes, a new Tool version
        is upserted with an updated JSON Schema.

        For example:

        ```python
        # Adding @tool on this function
        @humanloop_client.tool(path="calculator")
        def calculator(a: int, b: Optional[int]) -> int:
            \"\"\"Add two numbers together.\"\"\"
            return a + b

        # Creates a Tool with this JSON Schema:
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

        The return value of the decorated function must be JSON serializable.

        If the function raises an exception, the created Log will have `output`
        set to null, and the `error` field populated.

        :param path: The path of the File in the Humanloop workspace.

        :param setup_values: Values needed to setup the Tool, defined in [JSON Schema](https://json-schema.org/)

        :param attributes: Additional fields to describe the Tool. Helpful to separate Tool versions from each other with details on how they were created or used.
        """
        return tool_decorator_factory(
            opentelemetry_tracer=self._opentelemetry_tracer,
            path=path,
            attributes=attributes,
            setup_values=setup_values,
        )

    def flow(
        self,
        *,
        path: str,
        attributes: Optional[dict[str, Any]] = None,
    ):
        """Trace SDK logging calls through [Flows](https://humanloop.com/docs/explanation/flows).

        Use it as the entrypoint of your LLM feature. Logging calls like `prompts.call(...)`,
        `tools.call(...)`, or other Humanloop decorators will be automatically added to the trace.

        For example:

        ```python
        @prompt(template="You are an assistant on the following topics: {{topics}}.")
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
        def agent():
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

        Each call to agent will create a trace corresponding to the conversation
        session. Multiple Prompt Logs will be created as the LLM is called. They
        will be added to the trace, allowing you to see the whole conversation
        in the UI.

        If the function returns a ChatMessage-like object, the Log will
        populate the `output_message` field. Otherwise, it will serialize
        the return value and populate the `output` field.

        If an exception is raised, the output fields will be set to None
        and the error message will be set in the Log's `error` field.

        :param path: The path to the Flow. If not provided, the function name
            will be used as the path and the File will be created in the root
            of your organization workspace.

        :param attributes: Additional fields to describe the Flow. Helpful to separate Flow versions from each other with details on how they were created or used.
        """
        return flow_decorator_factory(
            client=self,
            opentelemetry_tracer=self._opentelemetry_tracer,
            path=path,
            attributes=attributes,
        )


class AsyncHumanloop(AsyncBaseHumanloop):
    """
    See docstring of AsyncBaseHumanloop.

    TODO: Add custom evaluation utilities for async case.
    """

    pass
