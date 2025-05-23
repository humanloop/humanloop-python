import logging
import os
import typing
from typing import Any, List, Optional, Sequence, Tuple

import httpx
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer

from humanloop.base_client import AsyncBaseHumanloop, BaseHumanloop
from humanloop.core.client_wrapper import SyncClientWrapper
from humanloop.decorators.flow import a_flow_decorator_factory as a_flow_decorator_factory
from humanloop.decorators.flow import flow_decorator_factory as flow_decorator_factory
from humanloop.decorators.prompt import a_prompt_decorator_factory, prompt_decorator_factory
from humanloop.decorators.tool import a_tool_decorator_factory as a_tool_decorator_factory
from humanloop.decorators.tool import tool_decorator_factory as tool_decorator_factory
from humanloop.environment import HumanloopEnvironment
from humanloop.evals import run_eval
from humanloop.evals.types import (
    DatasetEvalConfig,
    EvaluatorCheck,
    EvaluatorEvalConfig,
    FileEvalConfig,
)
from humanloop.evaluations.client import EvaluationsClient
from humanloop.otel import instrument_provider
from humanloop.otel.exporter import HumanloopSpanExporter
from humanloop.otel.processor import HumanloopSpanProcessor
from humanloop.overload import overload_client
from humanloop.prompt_utils import populate_template
from humanloop.prompts.client import PromptsClient
from humanloop.sync.file_syncer import DEFAULT_CACHE_SIZE, FileSyncer

logger = logging.getLogger("humanloop.sdk")


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
        file: FileEvalConfig,
        name: Optional[str],
        dataset: DatasetEvalConfig,
        evaluators: Optional[Sequence[EvaluatorEvalConfig]] = None,
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


class ExtendedPromptsClient(PromptsClient):
    """
    Adds utility for populating Prompt template inputs.
    """

    populate_template = staticmethod(populate_template)  # type: ignore [assignment]


class Humanloop(BaseHumanloop):
    """
    See docstring of :class:`BaseHumanloop`.

    This class extends the base client with custom evaluation utilities,
    decorators for declaring Files in code, and utilities for syncing
    files between Humanloop and local filesystem.
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
        use_local_files: bool = False,
        local_files_directory: str = "humanloop",
        cache_size: int = DEFAULT_CACHE_SIZE,
    ):
        """
        Extends the base client with custom evaluation utilities and
        decorators for declaring Files in code.

        The Humanloop SDK File decorators use OpenTelemetry internally.
        You can provide a TracerProvider and a Tracer to integrate
        with your existing telemetry system. If not provided,
        an internal TracerProvider will be used.

        Parameters
        ----------
        base_url: Optional base URL for the API
        environment: The environment to use (default: DEFAULT)
        api_key: Your Humanloop API key (default: from HUMANLOOP_API_KEY env var)
        timeout: Optional timeout for API requests
        follow_redirects: Whether to follow redirects
        httpx_client: Optional custom httpx client
        opentelemetry_tracer_provider: Optional tracer provider for telemetry
        opentelemetry_tracer: Optional tracer for telemetry
        use_local_files: Whether to use local files for prompts and agents
        local_files_directory: Base directory where local prompt and agent files are stored (default: "humanloop").
                      This is relative to the current working directory. For example:
                      - "humanloop" will look for files in "./humanloop/"
                      - "data/humanloop" will look for files in "./data/humanloop/"
                      When using paths in the API, they must be relative to this directory. For example,
                      if local_files_directory="humanloop" and you have a file at "humanloop/samples/test.prompt",
                      you would reference it as "samples/test" in your code.
        cache_size: Maximum number of files to cache when use_local_files is True (default: DEFAULT_CACHE_SIZE).
                   This parameter has no effect if use_local_files is False.
        """
        super().__init__(
            base_url=base_url,
            environment=environment,
            api_key=api_key,
            timeout=timeout,
            follow_redirects=follow_redirects,
            httpx_client=httpx_client,
        )

        self.use_local_files = use_local_files

        # Warn user if cache_size is non-default but use_local_files is False — has no effect and will therefore be ignored
        if not self.use_local_files and cache_size != DEFAULT_CACHE_SIZE:
            logger.warning(
                f"The specified cache_size={cache_size} will have no effect because use_local_files=False. "
                f"File caching is only active when local files are enabled."
            )

        # Check if cache_size is non-default but use_local_files is False
        self._file_syncer = FileSyncer(client=self, base_dir=local_files_directory, cache_size=cache_size)
        eval_client = ExtendedEvalsClient(client_wrapper=self._client_wrapper)
        eval_client.client = self
        self.evaluations = eval_client
        self.prompts = ExtendedPromptsClient(client_wrapper=self._client_wrapper)

        # Overload the .log method of the clients to be aware of Evaluation Context
        # and the @flow decorator providing the trace_id
        # Additionally, call and log methods are overloaded in the prompts and agents client to support the use of local files
        self.prompts = overload_client(
            client=self.prompts, file_syncer=self._file_syncer, use_local_files=self.use_local_files
        )
        self.agents = overload_client(
            client=self.agents, file_syncer=self._file_syncer, use_local_files=self.use_local_files
        )
        self.flows = overload_client(client=self.flows)
        self.tools = overload_client(client=self.tools)

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
            self._opentelemetry_tracer = self._tracer_provider.get_tracer("humanloop.sdk")
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

    def a_prompt(
        self,
        *,
        path: str,
    ):
        """Auto-instrument LLM providers and create [Prompt](https://humanloop.com/docs/explanation/prompts)
        Logs on Humanloop from them, for async functions.

        ```python
        @a_prompt(path="My Async Prompt")
        async def call_llm_async(messages):
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = await client.chat.completions.create(
                model="gpt-4o",
                temperature=0.8,
                frequency_penalty=0.5,
                max_tokens=200,
                messages=messages,
            )
            return response.choices[0].message.content

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
        Prompt version is created.

        :param path: The path where the Prompt is created. If not
            provided, the function name is used as the path and the File
            is created in the root of your Humanloop organization workspace.

        :param prompt_kernel: Attributes that define the Prompt. See `class:DecoratorPromptKernelRequestParams`
        """
        return a_prompt_decorator_factory(path=path)

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

    def a_tool(
        self,
        *,
        path: str,
        attributes: Optional[dict[str, Any]] = None,
        setup_values: Optional[dict[str, Any]] = None,
    ):
        """Manage async [Tool](https://humanloop.com/docs/explanation/tools) Files through code.

        The decorator inspects the wrapped async function's source code to infer the Tool's
        JSON Schema. If the function declaration changes, a new Tool version
        is upserted with an updated JSON Schema.

        For example:

        ```python
        # Adding @a_tool on this function
        @humanloop_client.a_tool(path="async_calculator")
        async def async_calculator(a: int, b: Optional[int]) -> int:
            \"\"\"Add two numbers together asynchronously.\"\"\"
            return a + b

        # Creates a Tool with this JSON Schema:
        {
            strict: True,
            function: {
                "name": "async_calculator",
                "description": "Add two numbers together asynchronously.",
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
        return a_tool_decorator_factory(
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

    def a_flow(
        self,
        *,
        path: str,
        attributes: Optional[dict[str, Any]] = None,
    ):
        """Trace SDK logging calls through [Flows](https://humanloop.com/docs/explanation/flows) for async functions.

        Use it as the entrypoint of your async LLM feature. Logging calls like `prompts.call(...)`,
        `tools.call(...)`, or other Humanloop decorators will be automatically added to the trace.

        For example:

        ```python
        @a_prompt(template="You are an assistant on the following topics: {{topics}}.")
        async def call_llm_async(messages):
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = await client.chat.completions.create(
                model="gpt-4o",
                temperature=0.8,
                frequency_penalty=0.5,
                max_tokens=200,
                messages=messages,
            )
            return response.choices[0].message.content

        @a_flow(attributes={"version": "v1"})
        async def async_agent():
            while True:
                messages = []
                user_input = input("You: ")
                if user_input == "exit":
                    break
                messages.append({"role": "user", "content": user_input})
                response = await call_llm_async(messages)
                messages.append({"role": "assistant", "content": response})
                print(f"Assistant: {response}")
        ```

        Each call to async_agent will create a trace corresponding to the conversation
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
        return a_flow_decorator_factory(
            client=self,
            opentelemetry_tracer=self._opentelemetry_tracer,
            path=path,
            attributes=attributes,
        )

    def pull(self, path: Optional[str] = None, environment: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """Pull Prompt and Agent files from Humanloop to local filesystem.

        This method will:
        1. Fetch Prompt and Agent files from your Humanloop workspace
        2. Save them to your local filesystem (directory specified by `local_files_directory`, default: "humanloop")
        3. Maintain the same directory structure as in Humanloop
        4. Add appropriate file extensions (`.prompt` or `.agent`)

        The path parameter can be used in two ways:
        - If it points to a specific file (e.g. "path/to/file.prompt" or "path/to/file.agent"), only that file will be pulled
        - If it points to a directory (e.g. "path/to/directory"), all Prompt and Agent files in that directory and its subdirectories will be pulled
        - If no path is provided, all Prompt and Agent files will be pulled

        The operation will overwrite existing files with the latest version from Humanloop
        but will not delete local files that don't exist in the remote workspace.

        Currently only supports syncing Prompt and Agent files. Other file types will be skipped.

        For example, with the default `local_files_directory="humanloop"`, files will be saved as:
        ```
        ./humanloop/
        ├── my_project/
        │   ├── prompts/
        │   │   ├── my_prompt.prompt
        │   │   └── nested/
        │   │       └── another_prompt.prompt
        │   └── agents/
        │       └── my_agent.agent
        └── another_project/
            └── prompts/
                └── other_prompt.prompt
        ```

        If you specify `local_files_directory="data/humanloop"`, files will be saved in ./data/humanloop/ instead.

        :param path: Optional path to either a specific file (e.g. "path/to/file.prompt") or a directory (e.g. "path/to/directory").
                    If not provided, all Prompt and Agent files will be pulled.
        :param environment: The environment to pull the files from.
        :return: Tuple of two lists:
             - First list contains paths of successfully synced files
             - Second list contains paths of files that failed to sync (due to API errors, missing content,
               or filesystem issues)
        :raises HumanloopRuntimeError: If there's an error communicating with the API
        """
        return self._file_syncer.pull(environment=environment, path=path)


class AsyncHumanloop(AsyncBaseHumanloop):
    """
    See docstring of AsyncBaseHumanloop.

    TODO: Add custom evaluation utilities for async case.
    """

    pass
