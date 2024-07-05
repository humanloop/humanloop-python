# Reference
## Prompts
<details><summary><code>client.prompts.<a href="src/humanloop/prompts/client.py">list_prompts</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of Prompts.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
response = client.prompts.list_prompts()
for item in response:
    yield item
# alternatively, you can paginate page-by-page
for page in response.iter_pages():
    yield page

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page number for pagination.
    
</dd>
</dl>

<dl>
<dd>

**size:** `typing.Optional[int]` — Page size for pagination. Number of Prompts to fetch.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Case-insensitive filter for Prompt name.
    
</dd>
</dl>

<dl>
<dd>

**user_filter:** `typing.Optional[str]` — Case-insensitive filter for users in the Prompt. This filter matches against both email address and name of users.
    
</dd>
</dl>

<dl>
<dd>

**sort_by:** `typing.Optional[ProjectSortBy]` — Field to sort Prompts by
    
</dd>
</dl>

<dl>
<dd>

**order:** `typing.Optional[SortOrder]` — Direction to sort by.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/humanloop/prompts/client.py">upsert</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a Prompt or update it with a new version if it already exists.

Prompts are identified by the `ID` or their `path`. The parameters (i.e. the prompt template, temperature, model etc.) determine the versions of the Prompt.

If you provide a commit message, then the new version will be committed;
otherwise it will be uncommitted. If you try to commit an already committed version,
an exception will be raised.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.prompts.upsert(
    model="model",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**model:** `str` — The model instance used, e.g. `gpt-4`. See [supported models](https://humanloop.com/docs/supported-models)
    
</dd>
</dl>

<dl>
<dd>

**path:** `typing.Optional[str]` — Path of the Prompt, including the name, which is used as a unique identifier.
    
</dd>
</dl>

<dl>
<dd>

**id:** `typing.Optional[str]` — ID for an existing Prompt to update.
    
</dd>
</dl>

<dl>
<dd>

**endpoint:** `typing.Optional[ModelEndpoints]` — The provider model endpoint used.
    
</dd>
</dl>

<dl>
<dd>

**template:** `typing.Optional[PromptRequestTemplate]` — For chat endpoint, provide a Chat template. For completion endpoint, provide a Prompt template. Input variables within the template should be specified with double curly bracket syntax: {{INPUT_NAME}}.
    
</dd>
</dl>

<dl>
<dd>

**provider:** `typing.Optional[ModelProviders]` — The company providing the underlying model service.
    
</dd>
</dl>

<dl>
<dd>

**max_tokens:** `typing.Optional[int]` — The maximum number of tokens to generate. Provide max_tokens=-1 to dynamically calculate the maximum number of tokens to generate given the length of the prompt
    
</dd>
</dl>

<dl>
<dd>

**temperature:** `typing.Optional[float]` — What sampling temperature to use when making a generation. Higher values means the model will be more creative.
    
</dd>
</dl>

<dl>
<dd>

**top_p:** `typing.Optional[float]` — An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
    
</dd>
</dl>

<dl>
<dd>

**stop:** `typing.Optional[PromptRequestStop]` — The string (or list of strings) after which the model will stop generating. The returned text will not contain the stop sequence.
    
</dd>
</dl>

<dl>
<dd>

**presence_penalty:** `typing.Optional[float]` — Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the generation so far.
    
</dd>
</dl>

<dl>
<dd>

**frequency_penalty:** `typing.Optional[float]` — Number between -2.0 and 2.0. Positive values penalize new tokens based on how frequently they appear in the generation so far.
    
</dd>
</dl>

<dl>
<dd>

**other:** `typing.Optional[typing.Dict[str, typing.Any]]` — Other parameter values to be passed to the provider call.
    
</dd>
</dl>

<dl>
<dd>

**seed:** `typing.Optional[int]` — If specified, model will make a best effort to sample deterministically, but it is not guaranteed.
    
</dd>
</dl>

<dl>
<dd>

**response_format:** `typing.Optional[ResponseFormat]` — The format of the response. Only `{"type": "json_object"}` is currently supported for chat.
    
</dd>
</dl>

<dl>
<dd>

**tools:** `typing.Optional[typing.Sequence[ToolFunction]]` — The tool specification that the model can choose to call if Tool calling is supported.
    
</dd>
</dl>

<dl>
<dd>

**linked_tools:** `typing.Optional[typing.Sequence[str]]` — The IDs of the Tools in your organization that the model can choose to call if Tool calling is supported. The default deployed version of that tool is called.
    
</dd>
</dl>

<dl>
<dd>

**commit_message:** `typing.Optional[str]` — Message describing the changes made.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/humanloop/prompts/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Retrieve the Prompt with the given ID.

By default, the deployed version of the Prompt is returned. Use the query parameters
`version_id` or `environment` to target a specific version of the Prompt.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.prompts.get(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Prompt.
    
</dd>
</dl>

<dl>
<dd>

**version_id:** `typing.Optional[str]` — A specific Version ID of the Prompt to retrieve.
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[str]` — Name of the Environment to retrieve a deployed Version from.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/humanloop/prompts/client.py">delete</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete the Prompt with the given ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.prompts.delete(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Prompt.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/humanloop/prompts/client.py">move</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Move the Prompt to a different path or change the name.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.prompts.move(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Prompt.
    
</dd>
</dl>

<dl>
<dd>

**path:** `typing.Optional[str]` — Path of the Prompt including the Prompt name, which is used as a unique identifier.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Name of the Prompt.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/humanloop/prompts/client.py">listversions</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of all the versions of a Prompt.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.prompts.listversions(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Prompt.
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[VersionStatus]` — Filter versions by status: 'uncommitted', 'committed'. If no status is provided, all versions are returned.
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[str]` — Name of the environment to filter versions by. If no environment is provided, all versions are returned.
    
</dd>
</dl>

<dl>
<dd>

**evaluator_aggregates:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/humanloop/prompts/client.py">commit</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Commit the Prompt Version with the given ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.prompts.commit(
    id="id",
    version_id="version_id",
    commit_message="commit_message",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Prompt.
    
</dd>
</dl>

<dl>
<dd>

**version_id:** `str` — Unique identifier for the specific version of the Prompt.
    
</dd>
</dl>

<dl>
<dd>

**commit_message:** `str` — Message describing the changes made.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/humanloop/prompts/client.py">log</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Log to a Prompt.

You can use query parameters version_id, or environment, to target
an existing version of the Prompt. Otherwise the default deployed version will be chosen.

Instead of targeting an existing version explicitly, you can instead pass in
Prompt details in the request body. In this case, we will check if the details correspond
to an existing version of the Prompt, if not we will create a new version. This is helpful
in the case where you are storing or deriving your Prompt details in code.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.prompts.log()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**version_id:** `typing.Optional[str]` — A specific Version ID of the Prompt to log to.
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[str]` — Name of the Environment identifying a deployed version to log to.
    
</dd>
</dl>

<dl>
<dd>

**path:** `typing.Optional[str]` — Path of the Prompt, including the name, which is used as a unique identifier.
    
</dd>
</dl>

<dl>
<dd>

**id:** `typing.Optional[str]` — ID for an existing Prompt to update.
    
</dd>
</dl>

<dl>
<dd>

**output_message:** `typing.Optional[ChatMessage]` — The message returned by the provider.
    
</dd>
</dl>

<dl>
<dd>

**prompt_tokens:** `typing.Optional[int]` — Number of tokens in the prompt used to generate the output.
    
</dd>
</dl>

<dl>
<dd>

**output_tokens:** `typing.Optional[int]` — Number of tokens in the output generated by the model.
    
</dd>
</dl>

<dl>
<dd>

**prompt_cost:** `typing.Optional[float]` — Cost in dollars associated to the tokens in the prompt.
    
</dd>
</dl>

<dl>
<dd>

**output_cost:** `typing.Optional[float]` — Cost in dollars associated to the tokens in the output.
    
</dd>
</dl>

<dl>
<dd>

**finish_reason:** `typing.Optional[str]` — Reason the generation finished.
    
</dd>
</dl>

<dl>
<dd>

**prompt:** `typing.Optional[PromptKernelRequest]` — Details of your Prompt. A new Prompt version will be created if the provided details are new.
    
</dd>
</dl>

<dl>
<dd>

**messages:** `typing.Optional[typing.Sequence[ChatMessage]]` — The messages passed to the to provider chat endpoint.
    
</dd>
</dl>

<dl>
<dd>

**tool_choice:** `typing.Optional[PromptLogRequestToolChoice]` 

Controls how the model uses tools. The following options are supported: 
- `'none'` means the model will not call any tool and instead generates a message; this is the default when no tools are provided as part of the Prompt. 
- `'auto'` means the model can decide to call one or more of the provided tools; this is the default when tools are provided as part of the Prompt. 
- `'required'` means the model can decide to call one or more of the provided tools. 
- `{'type': 'function', 'function': {name': <TOOL_NAME>}}` forces the model to use the named function.
    
</dd>
</dl>

<dl>
<dd>

**output:** `typing.Optional[str]` — Generated output from your model for the provided inputs. Can be `None` if logging an error, or if creating a parent Log with the intention to populate it later.
    
</dd>
</dl>

<dl>
<dd>

**raw_output:** `typing.Optional[str]` — Raw output from the provider.
    
</dd>
</dl>

<dl>
<dd>

**created_at:** `typing.Optional[dt.datetime]` — User defined timestamp for when the log was created. 
    
</dd>
</dl>

<dl>
<dd>

**error:** `typing.Optional[str]` — Error message if the log is an error.
    
</dd>
</dl>

<dl>
<dd>

**provider_latency:** `typing.Optional[float]` — Duration of the logged event in seconds.
    
</dd>
</dl>

<dl>
<dd>

**provider_request:** `typing.Optional[typing.Dict[str, typing.Any]]` — Raw request sent to provider.
    
</dd>
</dl>

<dl>
<dd>

**provider_response:** `typing.Optional[typing.Dict[str, typing.Any]]` — Raw response received the provider.
    
</dd>
</dl>

<dl>
<dd>

**session_id:** `typing.Optional[str]` — Unique identifier for the Session to associate the Log to. Allows you to record multiple Logs to a Session (using an ID kept by your internal systems) by passing the same `session_id` in subsequent log requests. 
    
</dd>
</dl>

<dl>
<dd>

**parent_id:** `typing.Optional[str]` — Unique identifier for the parent Log in a Session. Should only be provided if `session_id` is provided. If provided, the Log will be nested under the parent Log within the Session.
    
</dd>
</dl>

<dl>
<dd>

**inputs:** `typing.Optional[typing.Dict[str, typing.Any]]` — The inputs passed to the prompt template.
    
</dd>
</dl>

<dl>
<dd>

**source:** `typing.Optional[str]` — Identifies where the model was called from.
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` — Any additional metadata to record.
    
</dd>
</dl>

<dl>
<dd>

**save:** `typing.Optional[bool]` — Whether the request/response payloads will be stored on Humanloop.
    
</dd>
</dl>

<dl>
<dd>

**source_datapoint_id:** `typing.Optional[str]` — Unique identifier for the Datapoint that this Log is derived from. This can be used by Humanloop to associate Logs to Evaluations. If provided, Humanloop will automatically associate this Log to Evaluations that require a Log for this Datapoint-Version pair.
    
</dd>
</dl>

<dl>
<dd>

**batches:** `typing.Optional[typing.Sequence[str]]` — Array of Batch Ids that this log is part of. Batches are used to group Logs together for offline Evaluations
    
</dd>
</dl>

<dl>
<dd>

**user:** `typing.Optional[str]` — End-user ID related to the Log.
    
</dd>
</dl>

<dl>
<dd>

**prompt_log_request_environment:** `typing.Optional[str]` — The name of the Environment the Log is associated to.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/humanloop/prompts/client.py">call</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Call a Prompt.

Calling a Prompt subsequently calls the model provider before logging
the data to Humanloop.

You can use query parameters version_id, or environment, to target
an existing version of the Prompt. Otherwise the default deployed version will be chosen.

Instead of targeting an existing version explicitly, you can instead pass in
Prompt details in the request body. In this case, we will check if the details correspond
to an existing version of the Prompt, if not we will create a new version. This is helpful
in the case where you are storing or deriving your Prompt details in code.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.prompts.call()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**version_id:** `typing.Optional[str]` — A specific Version ID of the Prompt to log to.
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[str]` — Name of the Environment identifying a deployed version to log to.
    
</dd>
</dl>

<dl>
<dd>

**path:** `typing.Optional[str]` — Path of the Prompt, including the name, which is used as a unique identifier.
    
</dd>
</dl>

<dl>
<dd>

**id:** `typing.Optional[str]` — ID for an existing Prompt to update.
    
</dd>
</dl>

<dl>
<dd>

**prompt:** `typing.Optional[PromptKernelRequest]` — Details of your Prompt. A new Prompt version will be created if the provided details are new.
    
</dd>
</dl>

<dl>
<dd>

**messages:** `typing.Optional[typing.Sequence[ChatMessage]]` — The messages passed to the to provider chat endpoint.
    
</dd>
</dl>

<dl>
<dd>

**tool_choice:** `typing.Optional[PromptCallRequestToolChoice]` 

Controls how the model uses tools. The following options are supported: 
- `'none'` means the model will not call any tool and instead generates a message; this is the default when no tools are provided as part of the Prompt. 
- `'auto'` means the model can decide to call one or more of the provided tools; this is the default when tools are provided as part of the Prompt. 
- `'required'` means the model can decide to call one or more of the provided tools. 
- `{'type': 'function', 'function': {name': <TOOL_NAME>}}` forces the model to use the named function.
    
</dd>
</dl>

<dl>
<dd>

**session_id:** `typing.Optional[str]` — Unique identifier for the Session to associate the Log to. Allows you to record multiple Logs to a Session (using an ID kept by your internal systems) by passing the same `session_id` in subsequent log requests. 
    
</dd>
</dl>

<dl>
<dd>

**parent_id:** `typing.Optional[str]` — Unique identifier for the parent Log in a Session. Should only be provided if `session_id` is provided. If provided, the Log will be nested under the parent Log within the Session.
    
</dd>
</dl>

<dl>
<dd>

**inputs:** `typing.Optional[typing.Dict[str, typing.Any]]` — The inputs passed to the prompt template.
    
</dd>
</dl>

<dl>
<dd>

**source:** `typing.Optional[str]` — Identifies where the model was called from.
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` — Any additional metadata to record.
    
</dd>
</dl>

<dl>
<dd>

**save:** `typing.Optional[bool]` — Whether the request/response payloads will be stored on Humanloop.
    
</dd>
</dl>

<dl>
<dd>

**source_datapoint_id:** `typing.Optional[str]` — Unique identifier for the Datapoint that this Log is derived from. This can be used by Humanloop to associate Logs to Evaluations. If provided, Humanloop will automatically associate this Log to Evaluations that require a Log for this Datapoint-Version pair.
    
</dd>
</dl>

<dl>
<dd>

**batches:** `typing.Optional[typing.Sequence[str]]` — Array of Batch Ids that this log is part of. Batches are used to group Logs together for offline Evaluations
    
</dd>
</dl>

<dl>
<dd>

**user:** `typing.Optional[str]` — End-user ID related to the Log.
    
</dd>
</dl>

<dl>
<dd>

**prompt_call_request_environment:** `typing.Optional[str]` — The name of the Environment the Log is associated to.
    
</dd>
</dl>

<dl>
<dd>

**provider_api_keys:** `typing.Optional[ProviderApiKeys]` — API keys required by each provider to make API calls. The API keys provided here are not stored by Humanloop. If not specified here, Humanloop will fall back to the key saved to your organization.
    
</dd>
</dl>

<dl>
<dd>

**num_samples:** `typing.Optional[int]` — The number of generations.
    
</dd>
</dl>

<dl>
<dd>

**stream:** `typing.Optional[bool]` — If true, tokens will be sent as data-only server-sent events. If num_samples > 1, samples are streamed back independently.
    
</dd>
</dl>

<dl>
<dd>

**return_inputs:** `typing.Optional[bool]` — Whether to return the inputs in the response. If false, the response will contain an empty dictionary under inputs. This is useful for reducing the size of the response. Defaults to true.
    
</dd>
</dl>

<dl>
<dd>

**logprobs:** `typing.Optional[int]` — Include the log probabilities of the top n tokens in the provider_response
    
</dd>
</dl>

<dl>
<dd>

**suffix:** `typing.Optional[str]` — The suffix that comes after a completion of inserted text. Useful for completions that act like inserts.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/humanloop/prompts/client.py">update_evaluators</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Activate and deactivate Evaluators for the Prompt.

An activated Evaluator will automatically be run on all new Logs
within the Prompt for monitoring purposes.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.prompts.update_evaluators(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**evaluators_to_activate:** `typing.Optional[
    typing.Sequence[
        EvaluatorActivationDeactivationRequestEvaluatorsToActivateItem
    ]
]` — Monitoring Evaluators to activate. These will be automatically run on new Logs.
    
</dd>
</dl>

<dl>
<dd>

**evaluators_to_deactivate:** `typing.Optional[
    typing.Sequence[
        EvaluatorActivationDeactivationRequestEvaluatorsToDeactivateItem
    ]
]` — Evaluators to deactivate. These will not be run on new Logs.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/humanloop/prompts/client.py">deploy</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Deploy Prompt to Environment.

Set the deployed Version for the specified Environment. This Prompt Version
will be used for calls made to the Prompt in this Environment.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.prompts.deploy(
    id="id",
    environment_id="environment_id",
    version_id="version_id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Prompt.
    
</dd>
</dl>

<dl>
<dd>

**environment_id:** `str` — Unique identifier for the Environment to deploy the Version to.
    
</dd>
</dl>

<dl>
<dd>

**version_id:** `str` — Unique identifier for the specific version of the Prompt.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/humanloop/prompts/client.py">remove_deployment</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Remove deployment of Prompt from Environment.

Remove the deployed Version for the specified Environment. This Prompt Version
will no longer be used for calls made to the Prompt in this Environment.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.prompts.remove_deployment(
    id="id",
    environment_id="environment_id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Prompt.
    
</dd>
</dl>

<dl>
<dd>

**environment_id:** `str` — Unique identifier for the Environment to remove the deployment from.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/humanloop/prompts/client.py">list_environments</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List all Environments and their deployed versions for the Prompt.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.prompts.list_environments(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Prompt.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Tools
<details><summary><code>client.tools.<a href="src/humanloop/tools/client.py">list</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of Tools.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
response = client.tools.list()
for item in response:
    yield item
# alternatively, you can paginate page-by-page
for page in response.iter_pages():
    yield page

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page offset for pagination.
    
</dd>
</dl>

<dl>
<dd>

**size:** `typing.Optional[int]` — Page size for pagination. Number of Tools to fetch.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Case-insensitive filter for Tool name.
    
</dd>
</dl>

<dl>
<dd>

**user_filter:** `typing.Optional[str]` — Case-insensitive filter for users in the Tool. This filter matches against both email address and name of users.
    
</dd>
</dl>

<dl>
<dd>

**sort_by:** `typing.Optional[ProjectSortBy]` — Field to sort Tools by
    
</dd>
</dl>

<dl>
<dd>

**order:** `typing.Optional[SortOrder]` — Direction to sort by.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.tools.<a href="src/humanloop/tools/client.py">upsert</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a Tool or update it with a new version if it already exists.

Tools are identified by the `ID` or their `path`. The name, description and parameters determine the versions of the Tool.

If you provide a commit message, then the new version will be committed;
otherwise it will be uncommitted. If you try to commit an already committed version,
an exception will be raised.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.tools.upsert()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**path:** `typing.Optional[str]` — Path of the Tool, including the name, which is used as a unique identifier.
    
</dd>
</dl>

<dl>
<dd>

**id:** `typing.Optional[str]` — ID for an existing Tool to update.
    
</dd>
</dl>

<dl>
<dd>

**function:** `typing.Optional[ToolFunction]` — Callable function specification of the Tool shown to the model for tool calling.
    
</dd>
</dl>

<dl>
<dd>

**source_code:** `typing.Optional[str]` — Code source of the Tool.
    
</dd>
</dl>

<dl>
<dd>

**setup_values:** `typing.Optional[typing.Dict[str, typing.Any]]` — Values needed to setup the Tool, defined in JSON Schema format: https://json-schema.org/
    
</dd>
</dl>

<dl>
<dd>

**tool_type:** `typing.Optional[FilesToolType]` — Type of Tool.
    
</dd>
</dl>

<dl>
<dd>

**commit_message:** `typing.Optional[str]` — Message describing the changes made.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.tools.<a href="src/humanloop/tools/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Retrieve the Tool with the given ID.

By default, the deployed version of the Tool is returned. Use the query parameters
`version_id` or `environment` to target a specific version of the Tool.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.tools.get(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Tool.
    
</dd>
</dl>

<dl>
<dd>

**version_id:** `typing.Optional[str]` — A specific Version ID of the Tool to retrieve.
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[str]` — Name of the Environment to retrieve a deployed Version from.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.tools.<a href="src/humanloop/tools/client.py">delete</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete the Tool with the given ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.tools.delete(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Tool.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.tools.<a href="src/humanloop/tools/client.py">move</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Move the Tool to a different path or change the name.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.tools.move(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Tool.
    
</dd>
</dl>

<dl>
<dd>

**path:** `typing.Optional[str]` — Path of the Tool including the Tool name, which is used as a unique identifier.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Name of the Tool, which is used as a unique identifier.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.tools.<a href="src/humanloop/tools/client.py">listversions</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of all the versions of a Tool.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.tools.listversions(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for the Tool.
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[VersionStatus]` — Filter versions by status: 'uncommitted', 'committed'. If no status is provided, all versions are returned.
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[str]` — Name of the environment to filter versions by. If no environment is provided, all versions are returned.
    
</dd>
</dl>

<dl>
<dd>

**evaluator_aggregates:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.tools.<a href="src/humanloop/tools/client.py">commit</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Commit the Tool Version with the given ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.tools.commit(
    id="id",
    version_id="version_id",
    commit_message="commit_message",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Tool.
    
</dd>
</dl>

<dl>
<dd>

**version_id:** `str` — Unique identifier for the specific version of the Tool.
    
</dd>
</dl>

<dl>
<dd>

**commit_message:** `str` — Message describing the changes made.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.tools.<a href="src/humanloop/tools/client.py">log</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Log to a Tool.

You can use query parameters version_id, or environment, to target
an existing version of the Tool. Otherwise the default deployed version will be chosen.

Instead of targeting an existing version explicitly, you can instead pass in
Tool details in the request body. In this case, we will check if the details correspond
to an existing version of the Tool, if not we will create a new version. This is helpful
in the case where you are storing or deriving your Tool details in code.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.tools.log()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**version_id:** `typing.Optional[str]` — A specific Version ID of the Tool to log to.
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[str]` — Name of the Environment identifying a deployed version to log to.
    
</dd>
</dl>

<dl>
<dd>

**path:** `typing.Optional[str]` — Path of the Tool, including the name, which is used as a unique identifier.
    
</dd>
</dl>

<dl>
<dd>

**id:** `typing.Optional[str]` — ID for an existing Tool to update.
    
</dd>
</dl>

<dl>
<dd>

**output:** `typing.Optional[str]` — Generated output from your model for the provided inputs. Can be `None` if logging an error, or if creating a parent Log with the intention to populate it later.
    
</dd>
</dl>

<dl>
<dd>

**raw_output:** `typing.Optional[str]` — Raw output from the provider.
    
</dd>
</dl>

<dl>
<dd>

**created_at:** `typing.Optional[dt.datetime]` — User defined timestamp for when the log was created. 
    
</dd>
</dl>

<dl>
<dd>

**error:** `typing.Optional[str]` — Error message if the log is an error.
    
</dd>
</dl>

<dl>
<dd>

**provider_latency:** `typing.Optional[float]` — Duration of the logged event in seconds.
    
</dd>
</dl>

<dl>
<dd>

**provider_request:** `typing.Optional[typing.Dict[str, typing.Any]]` — Raw request sent to provider.
    
</dd>
</dl>

<dl>
<dd>

**provider_response:** `typing.Optional[typing.Dict[str, typing.Any]]` — Raw response received the provider.
    
</dd>
</dl>

<dl>
<dd>

**session_id:** `typing.Optional[str]` — Unique identifier for the Session to associate the Log to. Allows you to record multiple Logs to a Session (using an ID kept by your internal systems) by passing the same `session_id` in subsequent log requests. 
    
</dd>
</dl>

<dl>
<dd>

**parent_id:** `typing.Optional[str]` — Unique identifier for the parent Log in a Session. Should only be provided if `session_id` is provided. If provided, the Log will be nested under the parent Log within the Session.
    
</dd>
</dl>

<dl>
<dd>

**inputs:** `typing.Optional[typing.Dict[str, typing.Any]]` — The inputs passed to the prompt template.
    
</dd>
</dl>

<dl>
<dd>

**source:** `typing.Optional[str]` — Identifies where the model was called from.
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` — Any additional metadata to record.
    
</dd>
</dl>

<dl>
<dd>

**save:** `typing.Optional[bool]` — Whether the request/response payloads will be stored on Humanloop.
    
</dd>
</dl>

<dl>
<dd>

**source_datapoint_id:** `typing.Optional[str]` — Unique identifier for the Datapoint that this Log is derived from. This can be used by Humanloop to associate Logs to Evaluations. If provided, Humanloop will automatically associate this Log to Evaluations that require a Log for this Datapoint-Version pair.
    
</dd>
</dl>

<dl>
<dd>

**batches:** `typing.Optional[typing.Sequence[str]]` — Array of Batch Ids that this log is part of. Batches are used to group Logs together for offline Evaluations
    
</dd>
</dl>

<dl>
<dd>

**user:** `typing.Optional[str]` — End-user ID related to the Log.
    
</dd>
</dl>

<dl>
<dd>

**tool_log_request_environment:** `typing.Optional[str]` — The name of the Environment the Log is associated to.
    
</dd>
</dl>

<dl>
<dd>

**tool:** `typing.Optional[ToolKernelRequest]` — Details of your Tool. A new Tool version will be created if the provided details are new.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.tools.<a href="src/humanloop/tools/client.py">update_evaluators</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Activate and deactivate Evaluators for the Tool.

An activated Evaluator will automatically be run on all new Logs
within the Tool for monitoring purposes.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.tools.update_evaluators(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**evaluators_to_activate:** `typing.Optional[
    typing.Sequence[
        EvaluatorActivationDeactivationRequestEvaluatorsToActivateItem
    ]
]` — Monitoring Evaluators to activate. These will be automatically run on new Logs.
    
</dd>
</dl>

<dl>
<dd>

**evaluators_to_deactivate:** `typing.Optional[
    typing.Sequence[
        EvaluatorActivationDeactivationRequestEvaluatorsToDeactivateItem
    ]
]` — Evaluators to deactivate. These will not be run on new Logs.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.tools.<a href="src/humanloop/tools/client.py">deploy</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Deploy Tool to Environment.

Set the deployed Version for the specified Environment. This Tool Version
will be used for calls made to the Tool in this Environment.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.tools.deploy(
    id="id",
    environment_id="environment_id",
    version_id="version_id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Tool.
    
</dd>
</dl>

<dl>
<dd>

**environment_id:** `str` — Unique identifier for the Environment to deploy the Version to.
    
</dd>
</dl>

<dl>
<dd>

**version_id:** `str` — Unique identifier for the specific version of the Tool.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.tools.<a href="src/humanloop/tools/client.py">remove_deployment</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Remove deployment of Tool from Environment.

Remove the deployed Version for the specified Environment. This Tool Version
will no longer be used for calls made to the Tool in this Environment.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.tools.remove_deployment(
    id="id",
    environment_id="environment_id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Tool.
    
</dd>
</dl>

<dl>
<dd>

**environment_id:** `str` — Unique identifier for the Environment to remove the deployment from.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.tools.<a href="src/humanloop/tools/client.py">list_environments</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List all Environments and their deployed versions for the Tool.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.tools.list_environments(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Tool.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Datasets
<details><summary><code>client.datasets.<a href="src/humanloop/datasets/client.py">list</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of Datasets.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
response = client.datasets.list()
for item in response:
    yield item
# alternatively, you can paginate page-by-page
for page in response.iter_pages():
    yield page

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page offset for pagination.
    
</dd>
</dl>

<dl>
<dd>

**size:** `typing.Optional[int]` — Page size for pagination. Number of Datasets to fetch.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Case-insensitive filter for Dataset name.
    
</dd>
</dl>

<dl>
<dd>

**user_filter:** `typing.Optional[str]` — Case-insensitive filter for users in the Dataset. This filter matches against both email address and name of users.
    
</dd>
</dl>

<dl>
<dd>

**sort_by:** `typing.Optional[ProjectSortBy]` — Field to sort Datasets by
    
</dd>
</dl>

<dl>
<dd>

**order:** `typing.Optional[SortOrder]` — Direction to sort by.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/humanloop/datasets/client.py">upsert</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a Dataset or update it with a new version if it already exists.

Datasets are identified by the `ID` or their `path`. The datapoints determine the versions of the Dataset.

By default, the new Dataset version will be set to the list of Datapoints provided in
the request. You can also create a new version by adding or removing Datapoints from an existing version
by specifying `action` as `add` or `remove` respectively. In this case, you may specify
the `version_id` or `environment` query parameters to identify the existing version to base
the new version on. If neither is provided, the default deployed version will be used.

If you provide a commit message, then the new version will be committed;
otherwise it will be uncommitted. If you try to commit an already committed version,
an exception will be raised.

Humanloop also deduplicates Datapoints. If you try to add a Datapoint that already
exists, it will be ignored. If you intentionally want to add a duplicate Datapoint,
you can add a unique identifier to the Datapoint's inputs such as `{_dedupe_id: <unique ID>}`.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop import CreateDatapointRequest
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.datasets.upsert(
    datapoints=[CreateDatapointRequest()],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**datapoints:** `typing.Sequence[CreateDatapointRequest]` — The Datapoints to create this Dataset version with. Modify the `action` field to determine how these Datapoints are used.
    
</dd>
</dl>

<dl>
<dd>

**version_id:** `typing.Optional[str]` — ID of the specific Dataset version to base the created Version on. Only used when `action` is `"add"` or `"remove"`.
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[str]` — Name of the Environment identifying a deployed Version to base the created Version on. Only used when `action` is `"add"` or `"remove"`.
    
</dd>
</dl>

<dl>
<dd>

**path:** `typing.Optional[str]` — Path of the Dataset, including the name, which is used as a unique identifier.
    
</dd>
</dl>

<dl>
<dd>

**id:** `typing.Optional[str]` — ID for an existing Dataset to update.
    
</dd>
</dl>

<dl>
<dd>

**action:** `typing.Optional[UpdateDatesetAction]` 

The action to take with the provided Datapoints.

 - If `"set"`, the created version will only contain the Datapoints provided in this request. 
 - If `"add"`, the created version will contain the Datapoints provided in this request in addition to the Datapoints in the target version. 
 - If `"remove"`, the created version will contain the Datapoints in the target version except for the Datapoints provided in this request. 

If `"add"` or `"remove"`, one of the `version_id` or `environment` query parameters may be provided.
    
</dd>
</dl>

<dl>
<dd>

**commit_message:** `typing.Optional[str]` — Message describing the changes made. If provided, a committed version of the Dataset is created. Otherwise, an uncommitted version is created.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/humanloop/datasets/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Retrieve the Dataset with the given ID.

Unless `include_datapoints` is set to `true`, the response will not include
the Datapoints.
Use the List Datapoints endpoint (`GET /{id}/datapoints`) to efficiently
retrieve Datapoints for a large Dataset.

By default, the deployed version of the Dataset is returned. Use the query parameters
`version_id` or `environment` to target a specific version of the Dataset.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.datasets.get(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Dataset.
    
</dd>
</dl>

<dl>
<dd>

**version_id:** `typing.Optional[str]` — A specific Version ID of the Dataset to retrieve.
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[str]` — Name of the Environment to retrieve a deployed Version from.
    
</dd>
</dl>

<dl>
<dd>

**include_datapoints:** `typing.Optional[bool]` — If set to `true`, include all Datapoints in the response. Defaults to `false`. Consider using the paginated List Datapoints endpoint instead.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/humanloop/datasets/client.py">delete</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete the Dataset with the given ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.datasets.delete(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Dataset.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/humanloop/datasets/client.py">move</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update the Dataset with the given ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.datasets.move(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Dataset.
    
</dd>
</dl>

<dl>
<dd>

**path:** `typing.Optional[str]` — Path of the Dataset including the Dataset name, which is used as a unique identifier.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Name of the Dataset, which is used as a unique identifier.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/humanloop/datasets/client.py">listdatapoints</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List all Datapoints for the Dataset with the given ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
response = client.datasets.listdatapoints(
    id="id",
)
for item in response:
    yield item
# alternatively, you can paginate page-by-page
for page in response.iter_pages():
    yield page

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Dataset.
    
</dd>
</dl>

<dl>
<dd>

**version_id:** `typing.Optional[str]` — A specific Version ID of the Dataset to retrieve.
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[str]` — Name of the Environment to retrieve a deployed Version from.
    
</dd>
</dl>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page number for pagination.
    
</dd>
</dl>

<dl>
<dd>

**size:** `typing.Optional[int]` — Page size for pagination. Number of Datapoints to fetch.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/humanloop/datasets/client.py">listversions</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of the versions for a Dataset.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.datasets.listversions(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Dataset.
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[VersionStatus]` — Filter versions by status: 'uncommitted', 'committed'. If no status is provided, all versions are returned.
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[str]` — Name of the environment to filter versions by. If no environment is provided, all versions are returned.
    
</dd>
</dl>

<dl>
<dd>

**evaluation_aggregates:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/humanloop/datasets/client.py">commit</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Commit the Dataset Version with the given ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.datasets.commit(
    id="id",
    version_id="version_id",
    commit_message="commit_message",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Dataset.
    
</dd>
</dl>

<dl>
<dd>

**version_id:** `str` — Unique identifier for the specific version of the Dataset.
    
</dd>
</dl>

<dl>
<dd>

**commit_message:** `str` — Message describing the changes made.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/humanloop/datasets/client.py">createdatapointsfromlogs</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Add Datapoints from Logs to a Dataset.

This will create a new committed version of the Dataset with the Datapoints from the Logs.

If either `version_id` or `environment` is provided, the new version will be based on the specified version,
with the Datapoints from the Logs added to the existing Datapoints in the version.
If neither `version_id` nor `environment` is provided, the new version will be based on the version
of the Dataset that is deployed to the default Environment.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.datasets.createdatapointsfromlogs(
    id="id",
    log_ids=["log_ids"],
    commit_message="commit_message",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for the Dataset
    
</dd>
</dl>

<dl>
<dd>

**log_ids:** `typing.Sequence[str]` — List of Log IDs to create Datapoints from.
    
</dd>
</dl>

<dl>
<dd>

**commit_message:** `str` — Commit message for the new Dataset version.
    
</dd>
</dl>

<dl>
<dd>

**version_id:** `typing.Optional[str]` — ID of the specific Dataset version to base the created Version on.
    
</dd>
</dl>

<dl>
<dd>

**environment:** `typing.Optional[str]` — Name of the Environment identifying a deployed Version to base the created Version on.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/humanloop/datasets/client.py">deploy</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Deploy Dataset to Environment.

Set the deployed Version for the specified Environment.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.datasets.deploy(
    id="id",
    environment_id="environment_id",
    version_id="version_id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Dataset.
    
</dd>
</dl>

<dl>
<dd>

**environment_id:** `str` — Unique identifier for the Environment to deploy the Version to.
    
</dd>
</dl>

<dl>
<dd>

**version_id:** `str` — Unique identifier for the specific version of the Dataset.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/humanloop/datasets/client.py">remove_deployment</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Remove deployment of Dataset from Environment.

Remove the deployed Version for the specified Environment.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.datasets.remove_deployment(
    id="id",
    environment_id="environment_id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Dataset.
    
</dd>
</dl>

<dl>
<dd>

**environment_id:** `str` — Unique identifier for the Environment to remove the deployment from.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/humanloop/datasets/client.py">list_environments</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List all Environments and their deployed versions for the Dataset.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.datasets.list_environments(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Dataset.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Directories
<details><summary><code>client.directories.<a href="src/humanloop/directories/client.py">list</a>()</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Retrieve a list of all Directories.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.directories.list()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.directories.<a href="src/humanloop/directories/client.py">create</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a Directory.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.directories.create(
    name="name",
    parent_id="parent_id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**name:** `str` — Name of the directory to create.
    
</dd>
</dl>

<dl>
<dd>

**parent_id:** `str` — ID of the parent directory. Starts with `dir_`.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.directories.<a href="src/humanloop/directories/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Fetches a directory by ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.directories.get(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — String ID of directory. Starts with `dir_`.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.directories.<a href="src/humanloop/directories/client.py">delete</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete the Directory with the given ID.

The Directory must be empty (i.e. contain no Directories or Files).
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.directories.delete(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Directory. Starts with `dir_`.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.directories.<a href="src/humanloop/directories/client.py">update</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update the Directory with the given ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.directories.update(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Unique identifier for Directory. Starts with `dir_`.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Name to set for the directory.
    
</dd>
</dl>

<dl>
<dd>

**parent_id:** `typing.Optional[str]` — ID of the parent directory. Specify this to move directories. Starts with `dir_`.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Evaluations
<details><summary><code>client.evaluations.<a href="src/humanloop/evaluations/client.py">list</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List Evaluations for the given File.

Retrieve a list of Evaluations that evaluate versions of the specified File.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
response = client.evaluations.list(
    file_id="file_id",
)
for item in response:
    yield item
# alternatively, you can paginate page-by-page
for page in response.iter_pages():
    yield page

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**file_id:** `str` — Filter by File ID. If provided, only Evaluation for the specified File will be returned.
    
</dd>
</dl>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page number for pagination.
    
</dd>
</dl>

<dl>
<dd>

**size:** `typing.Optional[int]` — Page size for pagination. Number of Evaluations to fetch.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.evaluations.<a href="src/humanloop/evaluations/client.py">create</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create an Evaluation.

Create a new Evaluation by specifying the Dataset, Evaluatees, and Evaluators.
Humanloop will automatically start generating Logs and running Evaluators.

To keep updated on the progress of the Evaluation, you can poll the Evaluation
and check its status.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from humanloop import (
    EvaluateeRequest,
    EvaluationsDatasetRequest,
    EvaluationsRequest,
)
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.evaluations.create(
    dataset=EvaluationsDatasetRequest(
        version_id="version_id",
    ),
    evaluatees=[
        EvaluateeRequest(
            version_id="version_id",
        )
    ],
    evaluators=[
        EvaluationsRequest(
            version_id="version_id",
        )
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**dataset:** `EvaluationsDatasetRequest` — The Dataset Version to use in this Evaluation.
    
</dd>
</dl>

<dl>
<dd>

**evaluatees:** `typing.Sequence[EvaluateeRequest]` — Unique identifiers for the Prompt/Tool Versions to include in the Evaluation Report.
    
</dd>
</dl>

<dl>
<dd>

**evaluators:** `typing.Sequence[EvaluationsRequest]` — The Evaluators used to evaluate.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.evaluations.<a href="src/humanloop/evaluations/client.py">get</a>(...)</code></summary>
