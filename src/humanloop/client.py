from contextvars import ContextVar
import os
import typing
from typing import List, Optional, Sequence
from typing_extensions import Unpack

import httpx
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer

from humanloop.core.client_wrapper import SyncClientWrapper
from humanloop.decorators.types import DecoratorPromptKernelRequestParams
from humanloop.eval_utils.context import EVALUATION_CONTEXT_VARIABLE_NAME, EvaluationContext

from humanloop.eval_utils import log_with_evaluation_context, run_eval
from humanloop.eval_utils.types import Dataset, Evaluator, EvaluatorCheck, File

from humanloop.base_client import AsyncBaseHumanloop, BaseHumanloop
from humanloop.decorators.flow import flow as flow_decorator_factory
from humanloop.decorators.prompt import prompt as prompt_decorator_factory
from humanloop.decorators.tool import tool as tool_decorator_factory
from humanloop.environment import HumanloopEnvironment
from humanloop.evaluations.client import EvaluationsClient
from humanloop.otel import instrument_provider
from humanloop.otel.exporter import HumanloopSpanExporter
from humanloop.otel.processor import HumanloopSpanProcessor
from humanloop.prompt_utils import populate_template
from humanloop.prompts.client import PromptsClient
from humanloop.requests.flow_kernel_request import FlowKernelRequestParams
from humanloop.requests.tool_kernel_request import ToolKernelRequestParams


class ExtendedEvalsClient(EvaluationsClient):
    client: BaseHumanloop

    def __init__(
        self,
        *,
        client_wrapper: SyncClientWrapper,
        evaluation_context_variable: ContextVar[Optional[EvaluationContext]],
    ):
        super().__init__(client_wrapper=client_wrapper)
        self._evaluation_context_variable = evaluation_context_variable

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

        return run_eval(
            client=self.client,
            file=file,
            name=name,
            dataset=dataset,
            evaluators=evaluators,
            workers=workers,
            evaluation_context_variable=self._evaluation_context_variable,
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

        self.evaluation_context_variable: ContextVar[Optional[EvaluationContext]] = ContextVar(
            EVALUATION_CONTEXT_VARIABLE_NAME
        )

        eval_client = ExtendedEvalsClient(
            client_wrapper=self._client_wrapper,
            evaluation_context_variable=self.evaluation_context_variable,
        )
        eval_client.client = self
        self.evaluations = eval_client
        self.prompts = ExtendedPromptsClient(client_wrapper=self._client_wrapper)

        # Overload the .log method of the clients to be aware of Evaluation Context
        # TODO: Overload the log for Evaluators and Tools once run_id is added
        # to them.
        self.prompts = log_with_evaluation_context(
            client=self.prompts,
            evaluation_context_variable=self.evaluation_context_variable,
        )
        # self.evaluators = log_with_evaluation_context(client=self.evaluators)
        # self.tools = log_with_evaluation_context(client=self.tools)
        self.flows = log_with_evaluation_context(
            client=self.flows,
            evaluation_context_variable=self.evaluation_context_variable,
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

        if opentelemetry_tracer is None:
            self._opentelemetry_tracer = self._tracer_provider.get_tracer("humanloop.sdk")
        else:
            self._opentelemetry_tracer = opentelemetry_tracer

    def prompt(
        self,
        *,
        path: Optional[str] = None,
        **prompt_kernel: Unpack[DecoratorPromptKernelRequestParams],  # type: ignore
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
        ```

        This will create a (Prompt)[https://humanloop.com/docs/explanation/prompts] with the following attributes:

        ```python
        {
            model: "gpt-4o",
            endpoint: "chat",
            template: "You are an assistant on the following topics: {{topics}}.",
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

        The decorated function should return a string or the output should be JSON serializable. If
        the output cannot be serialized, TypeError will be raised.

        If the function raises an exception, the log created by the function will have the output
        field set to None and the error field set to the string representation of the exception.

        :param path: The path where the Prompt is created. If not
            provided, the function name is used as the path and the File
            is created in the root of your Humanloop organization workspace.

        :param prompt_kernel: Attributes that define the Prompt. See `class:DecoratorPromptKernelRequestParams`
        """
        return prompt_decorator_factory(
            opentelemetry_tracer=self._opentelemetry_tracer,
            path=path,
            **prompt_kernel,
        )

    def tool(
        self,
        *,
        path: Optional[str] = None,
        **tool_kernel: Unpack[ToolKernelRequestParams],  # type: ignore
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
        ```

        The decorated function should return a string or the output should be JSON serializable. If
        the output cannot be serialized, TypeError will be raised.

        If the function raises an exception, the log created by the function will have the output
        field set to None and the error field set to the string representation of the exception.

        :param path: The path to the Tool. If not provided, the function name
            will be used as the path and the File will be created in the root
            of your organization's workspace.

        :param tool_kernel: Attributes that define the Tool. See `class:ToolKernelRequestParams`
        """
        return tool_decorator_factory(
            opentelemetry_tracer=self._opentelemetry_tracer,
            path=path,
            **tool_kernel,
        )

    def flow(
        self,
        *,
        path: Optional[str] = None,
        **flow_kernel: Unpack[FlowKernelRequestParams],  # type: ignore
    ):
        """Decorator for declaring a [Flow](https://humanloop.com/docs/explanation/flows) in code.

        A [Flow](https://humanloop.com/docs/explanation/flows) decorator should be added
        at the entrypoint of your LLM feature. Call other functions decorated with
        Humanloop SDK decorators to create a Trace of Logs on Humanloop.

        Here's an example of declaring a [Flow](https://humanloop.com/docs/explanation/flows) in code:
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

        The decorated function should return a string or the output should be JSON serializable. If
        the output cannot be serialized, TypeError will be raised.

        If the function raises an exception, the log created by the function will have the output
        field set to None and the error field set to the string representation of the exception.

        :param path: The path to the Flow. If not provided, the function name
            will be used as the path and the File will be created in the root
            of your organization workspace.

        :param flow_kernel: Attributes that define the Flow. See `class:ToolKernelRequestParams`
        """
        return flow_decorator_factory(
            opentelemetry_tracer=self._opentelemetry_tracer,
            path=path,
            **flow_kernel,
        )


class AsyncHumanloop(AsyncBaseHumanloop):
    """
    See docstring of AsyncBaseHumanloop.

    TODO: Add custom evaluation utilities for async case.
    """

    pass
