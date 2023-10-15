<div align="center">

[![Visit Humanloop](https://raw.githubusercontent.com/humanloop/humanloop-python/HEAD/header.png)](https://humanloop.com)

# Humanloop


[![PyPI](https://img.shields.io/badge/PyPI-v0.5.17-blue)](https://pypi.org/project/humanloop/0.5.17)
[![GitHub last commit](https://img.shields.io/github/last-commit/humanloop/humanloop-python.svg)](https://github.com/humanloop/humanloop-python/commits)
[![README.md](https://img.shields.io/badge/README-Click%20Here-green)](https://github.com/humanloop/humanloop-python#readme)

</div>

## Table of Contents

<!-- toc -->

- [Requirements](#requirements)
- [Installing](#installing)
- [Getting Started](#getting-started)
- [Async](#async)
- [Streaming](#streaming)
- [Reference](#reference)
  * [`humanloop.chat`](#humanloopchat)
  * [`humanloop.chat_deployed`](#humanloopchat_deployed)
  * [`humanloop.chat_experiment`](#humanloopchat_experiment)
  * [`humanloop.chat_model_configuration`](#humanloopchat_model_configuration)
  * [`humanloop.complete`](#humanloopcomplete)
  * [`humanloop.complete_deployed`](#humanloopcomplete_deployed)
  * [`humanloop.complete_experiment`](#humanloopcomplete_experiment)
  * [`humanloop.complete_model_configuration`](#humanloopcomplete_model_configuration)
  * [`humanloop.datapoints.delete`](#humanloopdatapointsdelete)
  * [`humanloop.datapoints.get`](#humanloopdatapointsget)
  * [`humanloop.datapoints.update`](#humanloopdatapointsupdate)
  * [`humanloop.datasets.create`](#humanloopdatasetscreate)
  * [`humanloop.datasets.create_datapoint`](#humanloopdatasetscreate_datapoint)
  * [`humanloop.datasets.delete`](#humanloopdatasetsdelete)
  * [`humanloop.datasets.get`](#humanloopdatasetsget)
  * [`humanloop.datasets.list_all_for_project`](#humanloopdatasetslist_all_for_project)
  * [`humanloop.datasets.list_datapoints`](#humanloopdatasetslist_datapoints)
  * [`humanloop.datasets.update`](#humanloopdatasetsupdate)
  * [`humanloop.evaluations.create`](#humanloopevaluationscreate)
  * [`humanloop.evaluations.get`](#humanloopevaluationsget)
  * [`humanloop.evaluations.list_all_for_project`](#humanloopevaluationslist_all_for_project)
  * [`humanloop.evaluations.list_datapoints`](#humanloopevaluationslist_datapoints)
  * [`humanloop.evaluators.create`](#humanloopevaluatorscreate)
  * [`humanloop.evaluators.delete`](#humanloopevaluatorsdelete)
  * [`humanloop.evaluators.list`](#humanloopevaluatorslist)
  * [`humanloop.evaluators.update`](#humanloopevaluatorsupdate)
  * [`humanloop.experiments.create`](#humanloopexperimentscreate)
  * [`humanloop.experiments.delete`](#humanloopexperimentsdelete)
  * [`humanloop.experiments.list`](#humanloopexperimentslist)
  * [`humanloop.experiments.sample`](#humanloopexperimentssample)
  * [`humanloop.experiments.update`](#humanloopexperimentsupdate)
  * [`humanloop.feedback`](#humanloopfeedback)
  * [`humanloop.finetunes.create`](#humanloopfinetunescreate)
  * [`humanloop.finetunes.list_all_for_project`](#humanloopfinetuneslist_all_for_project)
  * [`humanloop.finetunes.summary`](#humanloopfinetunessummary)
  * [`humanloop.finetunes.update`](#humanloopfinetunesupdate)
  * [`humanloop.log`](#humanlooplog)
  * [`humanloop.logs.update`](#humanlooplogsupdate)
  * [`humanloop.logs.update_by_ref`](#humanlooplogsupdate_by_ref)
  * [`humanloop.model_configs.get`](#humanloopmodel_configsget)
  * [`humanloop.model_configs.register`](#humanloopmodel_configsregister)
  * [`humanloop.projects.create`](#humanloopprojectscreate)
  * [`humanloop.projects.create_feedback_type`](#humanloopprojectscreate_feedback_type)
  * [`humanloop.projects.deactivate_config`](#humanloopprojectsdeactivate_config)
  * [`humanloop.projects.deactivate_experiment`](#humanloopprojectsdeactivate_experiment)
  * [`humanloop.projects.delete_deployed_config`](#humanloopprojectsdelete_deployed_config)
  * [`humanloop.projects.deploy_config`](#humanloopprojectsdeploy_config)
  * [`humanloop.projects.export`](#humanloopprojectsexport)
  * [`humanloop.projects.get`](#humanloopprojectsget)
  * [`humanloop.projects.get_active_config`](#humanloopprojectsget_active_config)
  * [`humanloop.projects.list`](#humanloopprojectslist)
  * [`humanloop.projects.list_configs`](#humanloopprojectslist_configs)
  * [`humanloop.projects.list_deployed_configs`](#humanloopprojectslist_deployed_configs)
  * [`humanloop.projects.update`](#humanloopprojectsupdate)
  * [`humanloop.projects.update_feedback_types`](#humanloopprojectsupdate_feedback_types)
  * [`humanloop.sessions.create`](#humanloopsessionscreate)
  * [`humanloop.sessions.get`](#humanloopsessionsget)
  * [`humanloop.sessions.list`](#humanloopsessionslist)

<!-- tocstop -->

## Requirements

Python >=3.7

## Installing

```sh
pip install humanloop==0.5.17
```

## Getting Started

```python
from pprint import pprint
from humanloop import Humanloop, ApiException

humanloop = Humanloop(
    api_key="YOUR_API_KEY",
    openai_api_key="YOUR_OPENAI_API_KEY",
    anthropic_api_key="YOUR_ANTHROPIC_API_KEY",
)

try:
    # Chat
    chat_response = humanloop.chat(
        project="sdk-example",
        messages=[
            {
                "role": "user",
                "content": "Explain asynchronous programming.",
            }
        ],
        model_config={
            "model": "gpt-3.5-turbo",
            "max_tokens": -1,
            "temperature": 0.7,
            "chat_template": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant who replies in the style of {{persona}}.",
                },
            ],
        },
        inputs={
            "persona": "the pirate Blackbeard",
        },
        stream=False,
    )
    pprint(chat_response.body)
    pprint(chat_response.body["project_id"])
    pprint(chat_response.body["data"][0])
    pprint(chat_response.body["provider_responses"])
    pprint(chat_response.headers)
    pprint(chat_response.status)
    pprint(chat_response.round_trip_time)
except ApiException as e:
    print("Exception when calling .chat: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)

try:
    # Complete
    complete_response = humanloop.complete(
        project="sdk-example",
        inputs={
            "text": "Llamas that are well-socialized and trained to halter and lead after weaning and are very friendly and pleasant to be around. They are extremely curious and most will approach people easily. However, llamas that are bottle-fed or over-socialized and over-handled as youth will become extremely difficult to handle when mature, when they will begin to treat humans as they treat each other, which is characterized by bouts of spitting, kicking and neck wrestling.[33]",
        },
        model_config={
            "model": "gpt-3.5-turbo",
            "max_tokens": -1,
            "temperature": 0.7,
            "prompt_template": "Summarize this for a second-grade student:\n\nText:\n{{text}}\n\nSummary:\n",
        },
        stream=False,
    )
    pprint(complete_response.body)
    pprint(complete_response.body["project_id"])
    pprint(complete_response.body["data"][0])
    pprint(complete_response.body["provider_responses"])
    pprint(complete_response.headers)
    pprint(complete_response.status)
    pprint(complete_response.round_trip_time)
except ApiException as e:
    print("Exception when calling .complete: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)

try:
    # Feedback
    feedback_response = humanloop.feedback(
        type="rating",
        value="good",
        data_id="data_[...]",
        user="user@example.com",
    )
    pprint(feedback_response.body)
    pprint(feedback_response.headers)
    pprint(feedback_response.status)
    pprint(feedback_response.round_trip_time)
except ApiException as e:
    print("Exception when calling .feedback: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)

try:
    # Log
    log_response = humanloop.log(
        project="sdk-example",
        inputs={
            "text": "Llamas that are well-socialized and trained to halter and lead after weaning and are very friendly and pleasant to be around. They are extremely curious and most will approach people easily. However, llamas that are bottle-fed or over-socialized and over-handled as youth will become extremely difficult to handle when mature, when they will begin to treat humans as they treat each other, which is characterized by bouts of spitting, kicking and neck wrestling.[33]",
        },
        output="Llamas can be friendly and curious if they are trained to be around people, but if they are treated too much like pets when they are young, they can become difficult to handle when they grow up. This means they might spit, kick, and wrestle with their necks.",
        source="sdk",
        config={
            "model": "gpt-3.5-turbo",
            "max_tokens": -1,
            "temperature": 0.7,
            "prompt_template": "Summarize this for a second-grade student:\n\nText:\n{{text}}\n\nSummary:\n",
            "type": "model",
        },
    )
    pprint(log_response.body)
    pprint(log_response.headers)
    pprint(log_response.status)
    pprint(log_response.round_trip_time)
except ApiException as e:
    print("Exception when calling .log: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

## Async

`async` support is available by prepending `a` to any method.

```python
import asyncio
from pprint import pprint
from humanloop import Humanloop, ApiException

humanloop = Humanloop(
    api_key="YOUR_API_KEY",
    openai_api_key="YOUR_OPENAI_API_KEY",
    anthropic_api_key="YOUR_ANTHROPIC_API_KEY",
)


async def main():
    try:
        complete_response = await humanloop.acomplete(
            project="sdk-example",
            inputs={
                "text": "Llamas that are well-socialized and trained to halter and lead after weaning and are very friendly and pleasant to be around. They are extremely curious and most will approach people easily. However, llamas that are bottle-fed or over-socialized and over-handled as youth will become extremely difficult to handle when mature, when they will begin to treat humans as they treat each other, which is characterized by bouts of spitting, kicking and neck wrestling.[33]",
            },
            model_config={
                "model": "gpt-3.5-turbo",
                "max_tokens": -1,
                "temperature": 0.7,
                "prompt_template": "Summarize this for a second-grade student:\n\nText:\n{{text}}\n\nSummary:\n",
            },
            stream=False,
        )
        pprint(complete_response.body)
        pprint(complete_response.body["project_id"])
        pprint(complete_response.body["data"][0])
        pprint(complete_response.body["provider_responses"])
        pprint(complete_response.headers)
        pprint(complete_response.status)
        pprint(complete_response.round_trip_time)
    except ApiException as e:
        print("Exception when calling .complete: %s\n" % e)
        pprint(e.body)
        if e.status == 422:
            pprint(e.body["detail"])
        pprint(e.headers)
        pprint(e.status)
        pprint(e.reason)
        pprint(e.round_trip_time)


asyncio.run(main())
```


## Streaming

Streaming support is available by suffixing a `chat` or `complete` method with `_stream`.

```python
import asyncio
from humanloop import Humanloop

humanloop = Humanloop(
    api_key="YOUR_API_KEY",
    openai_api_key="YOUR_OPENAI_API_KEY",
    anthropic_api_key="YOUR_ANTHROPIC_API_KEY",
)


async def main():
    response = await humanloop.chat_stream(
        project="sdk-example",
        messages=[
            {
                "role": "user",
                "content": "Explain asynchronous programming.",
            }
        ],
        model_config={
            "model": "gpt-3.5-turbo",
            "max_tokens": -1,
            "temperature": 0.7,
            "chat_template": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant who replies in the style of {{persona}}.",
                },
            ],
        },
        inputs={
            "persona": "the pirate Blackbeard",
        },
    )
    async for token in response.content:
        print(token)


asyncio.run(main())
```


## Reference
### `humanloop.chat`

Get a chat response by providing details of the model configuration in the request.

#### 🛠️ Usage

```python
create_response = humanloop.chat(
    messages=[
        {
            "role": "user",
        }
    ],
    model_config={
        "model": "model_example",
        "max_tokens": -1,
        "temperature": 1,
        "top_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    },
    project="string_example",
    project_id="string_example",
    session_id="string_example",
    session_reference_id="string_example",
    parent_id="string_example",
    parent_reference_id="string_example",
    inputs={},
    source="string_example",
    metadata={},
    provider_api_keys={},
    num_samples=1,
    stream=False,
    user="string_example",
    tool_call="string_example",
)
```

#### ⚙️ Parameters

##### messages: List[[`ChatMessage`](./humanloop/type/chat_message.py)]

The messages passed to the to provider chat endpoint.

##### model_config: [`ModelConfigChatRequest`](./humanloop/type/model_config_chat_request.py)


The model configuration used to create a chat response.

##### project: `str`

Unique project name. If no project exists with this name, a new project will be created.

##### project_id: `str`

Unique ID of a project to associate to the log. Either this or `project` must be provided.

##### session_id: `str`

ID of the session to associate the datapoint.

##### session_reference_id: `str`

A unique string identifying the session to associate the datapoint to. Allows you to log multiple datapoints to a session (using an ID kept by your internal systems) by passing the same `session_reference_id` in subsequent log requests. Specify at most one of this or `session_id`.

##### parent_id: `str`

ID associated to the parent datapoint in a session.

##### parent_reference_id: `str`

A unique string identifying the previously-logged parent datapoint in a session. Allows you to log nested datapoints with your internal system IDs by passing the same reference ID as `parent_id` in a prior log request. Specify at most one of this or `parent_id`. Note that this cannot refer to a datapoint being logged in the same request.

##### inputs: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

The inputs passed to the prompt template.

##### source: `str`

Identifies where the model was called from.

##### metadata: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

Any additional metadata to record.

##### provider_api_keys: [`ProviderApiKeys`](./humanloop/type/provider_api_keys.py)


API keys required by each provider to make API calls. The API keys provided here are not stored by Humanloop. If not specified here, Humanloop will fall back to the key saved to your organization.

##### num_samples: `int`

The number of chat responses.

##### stream: `bool`

If true, tokens will be sent as data-only server-sent events. If num_samples > 1, samples are streamed back independently.

##### user: `str`

End-user ID passed through to provider call.

##### tool_call: Union[`str`, `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`]


Controls how the model uses tools - has the same behaviour as OpenAIs function_call parameter. The following options are supported: 'none' forces the model to not call a tool; the default when no tools are provided as part of the model config. 'auto' the model can decide to call one of the provided tools; the default when tools are provided as part of the model config. Providing {'name': <TOOL_NAME>} forces the model to use the provided tool of the same name.

#### ⚙️ Request Body

[`ChatRequest`](./humanloop/type/chat_request.py)
#### 🔄 Return

[ChatResponse](./humanloop/type/chat_response.py)

#### 🌐 Endpoint

`/chat` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.chat_deployed`

Get a chat response using the project's active deployment.  The active deployment can be a specific model configuration or an experiment.

#### 🛠️ Usage

```python
create_deployed_response = humanloop.chat_deployed(
    messages=[
        {
            "role": "user",
        }
    ],
    project="string_example",
    project_id="string_example",
    session_id="string_example",
    session_reference_id="string_example",
    parent_id="string_example",
    parent_reference_id="string_example",
    inputs={},
    source="string_example",
    metadata={},
    provider_api_keys={},
    num_samples=1,
    stream=False,
    user="string_example",
    tool_call="string_example",
    environment="string_example",
)
```

#### ⚙️ Parameters

##### messages: List[[`ChatMessage`](./humanloop/type/chat_message.py)]

The messages passed to the to provider chat endpoint.

##### project: `str`

Unique project name. If no project exists with this name, a new project will be created.

##### project_id: `str`

Unique ID of a project to associate to the log. Either this or `project` must be provided.

##### session_id: `str`

ID of the session to associate the datapoint.

##### session_reference_id: `str`

A unique string identifying the session to associate the datapoint to. Allows you to log multiple datapoints to a session (using an ID kept by your internal systems) by passing the same `session_reference_id` in subsequent log requests. Specify at most one of this or `session_id`.

##### parent_id: `str`

ID associated to the parent datapoint in a session.

##### parent_reference_id: `str`

A unique string identifying the previously-logged parent datapoint in a session. Allows you to log nested datapoints with your internal system IDs by passing the same reference ID as `parent_id` in a prior log request. Specify at most one of this or `parent_id`. Note that this cannot refer to a datapoint being logged in the same request.

##### inputs: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

The inputs passed to the prompt template.

##### source: `str`

Identifies where the model was called from.

##### metadata: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

Any additional metadata to record.

##### provider_api_keys: [`ProviderApiKeys`](./humanloop/type/provider_api_keys.py)


API keys required by each provider to make API calls. The API keys provided here are not stored by Humanloop. If not specified here, Humanloop will fall back to the key saved to your organization.

##### num_samples: `int`

The number of chat responses.

##### stream: `bool`

If true, tokens will be sent as data-only server-sent events. If num_samples > 1, samples are streamed back independently.

##### user: `str`

End-user ID passed through to provider call.

##### tool_call: Union[`str`, `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`]


Controls how the model uses tools - has the same behaviour as OpenAIs function_call parameter. The following options are supported: 'none' forces the model to not call a tool; the default when no tools are provided as part of the model config. 'auto' the model can decide to call one of the provided tools; the default when tools are provided as part of the model config. Providing {'name': <TOOL_NAME>} forces the model to use the provided tool of the same name.

##### environment: `str`

The environment name used to create a chat response. If not specified, the default environment will be used.

#### ⚙️ Request Body

[`ChatDeployedRequest`](./humanloop/type/chat_deployed_request.py)
#### 🔄 Return

[ChatResponse](./humanloop/type/chat_response.py)

#### 🌐 Endpoint

`/chat-deployed` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.chat_experiment`

Get a chat response for a specific experiment.

#### 🛠️ Usage

```python
create_experiment_response = humanloop.chat_experiment(
    messages=[
        {
            "role": "user",
        }
    ],
    experiment_id="string_example",
    project="string_example",
    project_id="string_example",
    session_id="string_example",
    session_reference_id="string_example",
    parent_id="string_example",
    parent_reference_id="string_example",
    inputs={},
    source="string_example",
    metadata={},
    provider_api_keys={},
    num_samples=1,
    stream=False,
    user="string_example",
    tool_call="string_example",
)
```

#### ⚙️ Parameters

##### messages: List[[`ChatMessage`](./humanloop/type/chat_message.py)]

The messages passed to the to provider chat endpoint.

##### experiment_id: `str`

If an experiment ID is provided a model configuration will be sampled from the experiments active model configurations.

##### project: `str`

Unique project name. If no project exists with this name, a new project will be created.

##### project_id: `str`

Unique ID of a project to associate to the log. Either this or `project` must be provided.

##### session_id: `str`

ID of the session to associate the datapoint.

##### session_reference_id: `str`

A unique string identifying the session to associate the datapoint to. Allows you to log multiple datapoints to a session (using an ID kept by your internal systems) by passing the same `session_reference_id` in subsequent log requests. Specify at most one of this or `session_id`.

##### parent_id: `str`

ID associated to the parent datapoint in a session.

##### parent_reference_id: `str`

A unique string identifying the previously-logged parent datapoint in a session. Allows you to log nested datapoints with your internal system IDs by passing the same reference ID as `parent_id` in a prior log request. Specify at most one of this or `parent_id`. Note that this cannot refer to a datapoint being logged in the same request.

##### inputs: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

The inputs passed to the prompt template.

##### source: `str`

Identifies where the model was called from.

##### metadata: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

Any additional metadata to record.

##### provider_api_keys: [`ProviderApiKeys`](./humanloop/type/provider_api_keys.py)


API keys required by each provider to make API calls. The API keys provided here are not stored by Humanloop. If not specified here, Humanloop will fall back to the key saved to your organization.

##### num_samples: `int`

The number of chat responses, where each chat response will use a model configuration sampled from the experiment.

##### stream: `bool`

If true, tokens will be sent as data-only server-sent events. If num_samples > 1, samples are streamed back independently.

##### user: `str`

End-user ID passed through to provider call.

##### tool_call: Union[`str`, `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`]


Controls how the model uses tools - has the same behaviour as OpenAIs function_call parameter. The following options are supported: 'none' forces the model to not call a tool; the default when no tools are provided as part of the model config. 'auto' the model can decide to call one of the provided tools; the default when tools are provided as part of the model config. Providing {'name': <TOOL_NAME>} forces the model to use the provided tool of the same name.

#### ⚙️ Request Body

[`ChatExperimentRequest`](./humanloop/type/chat_experiment_request.py)
#### 🔄 Return

[ChatResponse](./humanloop/type/chat_response.py)

#### 🌐 Endpoint

`/chat-experiment` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.chat_model_configuration`

Get chat response for a specific model configuration.

#### 🛠️ Usage

```python
create_model_config_response = humanloop.chat_model_configuration(
    messages=[
        {
            "role": "user",
        }
    ],
    model_config_id="string_example",
    project="string_example",
    project_id="string_example",
    session_id="string_example",
    session_reference_id="string_example",
    parent_id="string_example",
    parent_reference_id="string_example",
    inputs={},
    source="string_example",
    metadata={},
    provider_api_keys={},
    num_samples=1,
    stream=False,
    user="string_example",
    tool_call="string_example",
)
```

#### ⚙️ Parameters

##### messages: List[[`ChatMessage`](./humanloop/type/chat_message.py)]

The messages passed to the to provider chat endpoint.

##### model_config_id: `str`

Identifies the model configuration used to create a chat response.

##### project: `str`

Unique project name. If no project exists with this name, a new project will be created.

##### project_id: `str`

Unique ID of a project to associate to the log. Either this or `project` must be provided.

##### session_id: `str`

ID of the session to associate the datapoint.

##### session_reference_id: `str`

A unique string identifying the session to associate the datapoint to. Allows you to log multiple datapoints to a session (using an ID kept by your internal systems) by passing the same `session_reference_id` in subsequent log requests. Specify at most one of this or `session_id`.

##### parent_id: `str`

ID associated to the parent datapoint in a session.

##### parent_reference_id: `str`

A unique string identifying the previously-logged parent datapoint in a session. Allows you to log nested datapoints with your internal system IDs by passing the same reference ID as `parent_id` in a prior log request. Specify at most one of this or `parent_id`. Note that this cannot refer to a datapoint being logged in the same request.

##### inputs: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

The inputs passed to the prompt template.

##### source: `str`

Identifies where the model was called from.

##### metadata: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

Any additional metadata to record.

##### provider_api_keys: [`ProviderApiKeys`](./humanloop/type/provider_api_keys.py)


API keys required by each provider to make API calls. The API keys provided here are not stored by Humanloop. If not specified here, Humanloop will fall back to the key saved to your organization.

##### num_samples: `int`

The number of chat responses.

##### stream: `bool`

If true, tokens will be sent as data-only server-sent events. If num_samples > 1, samples are streamed back independently.

##### user: `str`

End-user ID passed through to provider call.

##### tool_call: Union[`str`, `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`]


Controls how the model uses tools - has the same behaviour as OpenAIs function_call parameter. The following options are supported: 'none' forces the model to not call a tool; the default when no tools are provided as part of the model config. 'auto' the model can decide to call one of the provided tools; the default when tools are provided as part of the model config. Providing {'name': <TOOL_NAME>} forces the model to use the provided tool of the same name.

#### ⚙️ Request Body

[`ChatModelConfigRequest`](./humanloop/type/chat_model_config_request.py)
#### 🔄 Return

[ChatResponse](./humanloop/type/chat_response.py)

#### 🌐 Endpoint

`/chat-model-config` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.complete`

Create a completion by providing details of the model configuration in the request.

#### 🛠️ Usage

```python
create_response = humanloop.complete(
    model_config={
        "model": "model_example",
        "max_tokens": -1,
        "temperature": 1,
        "top_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "prompt_template": "{{question}}",
    },
    project="string_example",
    project_id="string_example",
    session_id="string_example",
    session_reference_id="string_example",
    parent_id="string_example",
    parent_reference_id="string_example",
    inputs={},
    source="string_example",
    metadata={},
    provider_api_keys={},
    num_samples=1,
    logprobs=1,
    stream=False,
    suffix="string_example",
    user="string_example",
)
```

#### ⚙️ Parameters

##### model_config: [`ModelConfigCompletionRequest`](./humanloop/type/model_config_completion_request.py)


The model configuration used to generate.

##### project: `str`

Unique project name. If no project exists with this name, a new project will be created.

##### project_id: `str`

Unique ID of a project to associate to the log. Either this or `project` must be provided.

##### session_id: `str`

ID of the session to associate the datapoint.

##### session_reference_id: `str`

A unique string identifying the session to associate the datapoint to. Allows you to log multiple datapoints to a session (using an ID kept by your internal systems) by passing the same `session_reference_id` in subsequent log requests. Specify at most one of this or `session_id`.

##### parent_id: `str`

ID associated to the parent datapoint in a session.

##### parent_reference_id: `str`

A unique string identifying the previously-logged parent datapoint in a session. Allows you to log nested datapoints with your internal system IDs by passing the same reference ID as `parent_id` in a prior log request. Specify at most one of this or `parent_id`. Note that this cannot refer to a datapoint being logged in the same request.

##### inputs: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

The inputs passed to the prompt template.

##### source: `str`

Identifies where the model was called from.

##### metadata: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

Any additional metadata to record.

##### provider_api_keys: [`ProviderApiKeys`](./humanloop/type/provider_api_keys.py)


API keys required by each provider to make API calls. The API keys provided here are not stored by Humanloop. If not specified here, Humanloop will fall back to the key saved to your organization.

##### num_samples: `int`

The number of generations.

##### logprobs: `int`

Include the log probabilities of the top n tokens in the provider_response

##### stream: `bool`

If true, tokens will be sent as data-only server-sent events. If num_samples > 1, samples are streamed back independently.

##### suffix: `str`

The suffix that comes after a completion of inserted text. Useful for completions that act like inserts.

##### user: `str`

End-user ID passed through to provider call.

#### ⚙️ Request Body

[`CompletionRequest`](./humanloop/type/completion_request.py)
#### 🔄 Return

[CompletionResponse](./humanloop/type/completion_response.py)

#### 🌐 Endpoint

`/completion` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.complete_deployed`

Create a completion using the project's active deployment.  The active deployment can be a specific model configuration or an experiment.

#### 🛠️ Usage

```python
create_deployed_response = humanloop.complete_deployed(
    project="string_example",
    project_id="string_example",
    session_id="string_example",
    session_reference_id="string_example",
    parent_id="string_example",
    parent_reference_id="string_example",
    inputs={},
    source="string_example",
    metadata={},
    provider_api_keys={},
    num_samples=1,
    logprobs=1,
    stream=False,
    suffix="string_example",
    user="string_example",
    environment="string_example",
)
```

#### ⚙️ Parameters

##### project: `str`

Unique project name. If no project exists with this name, a new project will be created.

##### project_id: `str`

Unique ID of a project to associate to the log. Either this or `project` must be provided.

##### session_id: `str`

ID of the session to associate the datapoint.

##### session_reference_id: `str`

A unique string identifying the session to associate the datapoint to. Allows you to log multiple datapoints to a session (using an ID kept by your internal systems) by passing the same `session_reference_id` in subsequent log requests. Specify at most one of this or `session_id`.

##### parent_id: `str`

ID associated to the parent datapoint in a session.

##### parent_reference_id: `str`

A unique string identifying the previously-logged parent datapoint in a session. Allows you to log nested datapoints with your internal system IDs by passing the same reference ID as `parent_id` in a prior log request. Specify at most one of this or `parent_id`. Note that this cannot refer to a datapoint being logged in the same request.

##### inputs: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

The inputs passed to the prompt template.

##### source: `str`

Identifies where the model was called from.

##### metadata: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

Any additional metadata to record.

##### provider_api_keys: [`ProviderApiKeys`](./humanloop/type/provider_api_keys.py)


API keys required by each provider to make API calls. The API keys provided here are not stored by Humanloop. If not specified here, Humanloop will fall back to the key saved to your organization.

##### num_samples: `int`

The number of generations.

##### logprobs: `int`

Include the log probabilities of the top n tokens in the provider_response

##### stream: `bool`

If true, tokens will be sent as data-only server-sent events. If num_samples > 1, samples are streamed back independently.

##### suffix: `str`

The suffix that comes after a completion of inserted text. Useful for completions that act like inserts.

##### user: `str`

End-user ID passed through to provider call.

##### environment: `str`

The environment name used to create a chat response. If not specified, the default environment will be used.

#### ⚙️ Request Body

[`CompletionDeployedRequest`](./humanloop/type/completion_deployed_request.py)
#### 🔄 Return

[CompletionResponse](./humanloop/type/completion_response.py)

#### 🌐 Endpoint

`/completion-deployed` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.complete_experiment`

Create a completion for a specific experiment.

#### 🛠️ Usage

```python
create_experiment_response = humanloop.complete_experiment(
    experiment_id="string_example",
    project="string_example",
    project_id="string_example",
    session_id="string_example",
    session_reference_id="string_example",
    parent_id="string_example",
    parent_reference_id="string_example",
    inputs={},
    source="string_example",
    metadata={},
    provider_api_keys={},
    num_samples=1,
    logprobs=1,
    stream=False,
    suffix="string_example",
    user="string_example",
)
```

#### ⚙️ Parameters

##### experiment_id: `str`

If an experiment ID is provided a model configuration will be sampled from the experiments active model configurations.

##### project: `str`

Unique project name. If no project exists with this name, a new project will be created.

##### project_id: `str`

Unique ID of a project to associate to the log. Either this or `project` must be provided.

##### session_id: `str`

ID of the session to associate the datapoint.

##### session_reference_id: `str`

A unique string identifying the session to associate the datapoint to. Allows you to log multiple datapoints to a session (using an ID kept by your internal systems) by passing the same `session_reference_id` in subsequent log requests. Specify at most one of this or `session_id`.

##### parent_id: `str`

ID associated to the parent datapoint in a session.

##### parent_reference_id: `str`

A unique string identifying the previously-logged parent datapoint in a session. Allows you to log nested datapoints with your internal system IDs by passing the same reference ID as `parent_id` in a prior log request. Specify at most one of this or `parent_id`. Note that this cannot refer to a datapoint being logged in the same request.

##### inputs: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

The inputs passed to the prompt template.

##### source: `str`

Identifies where the model was called from.

##### metadata: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

Any additional metadata to record.

##### provider_api_keys: [`ProviderApiKeys`](./humanloop/type/provider_api_keys.py)


API keys required by each provider to make API calls. The API keys provided here are not stored by Humanloop. If not specified here, Humanloop will fall back to the key saved to your organization.

##### num_samples: `int`

The number of chat responses, where each chat response will use a model configuration sampled from the experiment.

##### logprobs: `int`

Include the log probabilities of the top n tokens in the provider_response

##### stream: `bool`

If true, tokens will be sent as data-only server-sent events. If num_samples > 1, samples are streamed back independently.

##### suffix: `str`

The suffix that comes after a completion of inserted text. Useful for completions that act like inserts.

##### user: `str`

End-user ID passed through to provider call.

#### ⚙️ Request Body

[`CompletionExperimentRequest`](./humanloop/type/completion_experiment_request.py)
#### 🔄 Return

[CompletionResponse](./humanloop/type/completion_response.py)

#### 🌐 Endpoint

`/completion-experiment` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.complete_model_configuration`

Create a completion for a specific model configuration.

#### 🛠️ Usage

```python
create_model_config_response = humanloop.complete_model_configuration(
    model_config_id="string_example",
    project="string_example",
    project_id="string_example",
    session_id="string_example",
    session_reference_id="string_example",
    parent_id="string_example",
    parent_reference_id="string_example",
    inputs={},
    source="string_example",
    metadata={},
    provider_api_keys={},
    num_samples=1,
    logprobs=1,
    stream=False,
    suffix="string_example",
    user="string_example",
)
```

#### ⚙️ Parameters

##### model_config_id: `str`

Identifies the model configuration used to create a chat response.

##### project: `str`

Unique project name. If no project exists with this name, a new project will be created.

##### project_id: `str`

Unique ID of a project to associate to the log. Either this or `project` must be provided.

##### session_id: `str`

ID of the session to associate the datapoint.

##### session_reference_id: `str`

A unique string identifying the session to associate the datapoint to. Allows you to log multiple datapoints to a session (using an ID kept by your internal systems) by passing the same `session_reference_id` in subsequent log requests. Specify at most one of this or `session_id`.

##### parent_id: `str`

ID associated to the parent datapoint in a session.

##### parent_reference_id: `str`

A unique string identifying the previously-logged parent datapoint in a session. Allows you to log nested datapoints with your internal system IDs by passing the same reference ID as `parent_id` in a prior log request. Specify at most one of this or `parent_id`. Note that this cannot refer to a datapoint being logged in the same request.

##### inputs: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

The inputs passed to the prompt template.

##### source: `str`

Identifies where the model was called from.

##### metadata: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

Any additional metadata to record.

##### provider_api_keys: [`ProviderApiKeys`](./humanloop/type/provider_api_keys.py)


API keys required by each provider to make API calls. The API keys provided here are not stored by Humanloop. If not specified here, Humanloop will fall back to the key saved to your organization.

##### num_samples: `int`

The number of generations.

##### logprobs: `int`

Include the log probabilities of the top n tokens in the provider_response

##### stream: `bool`

If true, tokens will be sent as data-only server-sent events. If num_samples > 1, samples are streamed back independently.

##### suffix: `str`

The suffix that comes after a completion of inserted text. Useful for completions that act like inserts.

##### user: `str`

End-user ID passed through to provider call.

#### ⚙️ Request Body

[`CompletionModelConfigRequest`](./humanloop/type/completion_model_config_request.py)
#### 🔄 Return

[CompletionResponse](./humanloop/type/completion_response.py)

#### 🌐 Endpoint

`/completion-model-config` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.datapoints.delete`

Delete a list of testsets by their IDs.

#### 🛠️ Usage

```python
humanloop.datapoints.delete(
    body=["datapoints_delete_request_example"],
)
```

#### ⚙️ Request Body

[`DatapointsDeleteRequest`](./humanloop/type/datapoints_delete_request.py)
#### 🌐 Endpoint

`/datapoints` `delete`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.datapoints.get`

Get a datapoint by ID.

#### 🛠️ Usage

```python
get_response = humanloop.datapoints.get(
    id="id_example",
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of datapoint. Starts with `evtc_`.

#### 🔄 Return

[DatapointResponse](./humanloop/type/datapoint_response.py)

#### 🌐 Endpoint

`/datapoints/{id}` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.datapoints.update`

Edit the input, messages and criteria fields of a datapoint. The fields passed in the request are the ones edited.  Passing `null` as a value for a field will delete that field. In order to signify not changing a field, it should be omitted from the request body.

#### 🛠️ Usage

```python
update_response = humanloop.datapoints.update(
    id="id_example",
    inputs={
        "key": "string_example",
    },
    messages=[
        {
            "role": "user",
        }
    ],
    target={
        "key": {},
    },
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of datapoint. Starts with `evtc_`.

##### inputs: [`UpdateDatapointRequestInputs`](./humanloop/type/update_datapoint_request_inputs.py)

##### messages: List[[`ChatMessage`](./humanloop/type/chat_message.py)]

The chat messages for this datapoint.

##### target: [`UpdateDatapointRequestTarget`](./humanloop/type/update_datapoint_request_target.py)

#### ⚙️ Request Body

[`UpdateDatapointRequest`](./humanloop/type/update_datapoint_request.py)
#### 🔄 Return

[DatapointResponse](./humanloop/type/datapoint_response.py)

#### 🌐 Endpoint

`/datapoints/{id}` `patch`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.datasets.create`

Create a new dataset for a project.

#### 🛠️ Usage

```python
create_response = humanloop.datasets.create(
    description="string_example",
    name="string_example",
    project_id="project_id_example",
)
```

#### ⚙️ Parameters

##### description: `str`

The description of the dataset.

##### name: `str`

The name of the dataset.

##### project_id: `str`

#### ⚙️ Request Body

[`CreateDatasetRequest`](./humanloop/type/create_dataset_request.py)
#### 🔄 Return

[DatasetResponse](./humanloop/type/dataset_response.py)

#### 🌐 Endpoint

`/projects/{project_id}/datasets` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.datasets.create_datapoint`

Create a new testcase for a testset.

#### 🛠️ Usage

```python
create_datapoint_response = humanloop.datasets.create_datapoint(
    body={
        "log_ids": ["log_ids_example"],
    },
    dataset_id="dataset_id_example",
    log_ids=["string_example"],
    inputs={
        "key": "string_example",
    },
    messages=[
        {
            "role": "user",
        }
    ],
    target={
        "key": {},
    },
)
```

#### ⚙️ Parameters

##### dataset_id: `str`

String ID of dataset. Starts with `evts_`.

##### requestBody: [`DatasetsCreateDatapointRequest`](./humanloop/type/datasets_create_datapoint_request.py)

#### 🔄 Return

[DatasetsCreateDatapointResponse](./humanloop/type/datasets_create_datapoint_response.py)

#### 🌐 Endpoint

`/datasets/{dataset_id}/datapoints` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.datasets.delete`

Delete a dataset by ID.

#### 🛠️ Usage

```python
delete_response = humanloop.datasets.delete(
    id="id_example",
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of dataset. Starts with `evts_`.

#### 🔄 Return

[DatasetResponse](./humanloop/type/dataset_response.py)

#### 🌐 Endpoint

`/datasets/{id}` `delete`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.datasets.get`

Get a single dataset by ID.

#### 🛠️ Usage

```python
get_response = humanloop.datasets.get(
    id="id_example",
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of dataset. Starts with `evts_`.

#### 🔄 Return

[DatasetResponse](./humanloop/type/dataset_response.py)

#### 🌐 Endpoint

`/datasets/{id}` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.datasets.list_all_for_project`

Get all datasets for a project.

#### 🛠️ Usage

```python
list_all_for_project_response = humanloop.datasets.list_all_for_project(
    project_id="project_id_example",
)
```

#### ⚙️ Parameters

##### project_id: `str`

#### 🔄 Return

[DatasetsListAllForProjectResponse](./humanloop/type/datasets_list_all_for_project_response.py)

#### 🌐 Endpoint

`/projects/{project_id}/datasets` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.datasets.list_datapoints`

Get datapoints for a dataset.

#### 🛠️ Usage

```python
list_datapoints_response = humanloop.datasets.list_datapoints(
    dataset_id="dataset_id_example",
    page=0,
    size=50,
)
```

#### ⚙️ Parameters

##### dataset_id: `str`

String ID of dataset. Starts with `evts_`.

##### page: `int`

##### size: `int`

#### 🌐 Endpoint

`/datasets/{dataset_id}/datapoints` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.datasets.update`

Update a testset by ID.

#### 🛠️ Usage

```python
update_response = humanloop.datasets.update(
    id="id_example",
    description="string_example",
    name="string_example",
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of testset. Starts with `evts_`.

##### description: `str`

The description of the dataset.

##### name: `str`

The name of the dataset.

#### ⚙️ Request Body

[`UpdateDatasetRequest`](./humanloop/type/update_dataset_request.py)
#### 🔄 Return

[DatasetResponse](./humanloop/type/dataset_response.py)

#### 🌐 Endpoint

`/datasets/{id}` `patch`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.evaluations.create`

Create an evaluation.

#### 🛠️ Usage

```python
create_response = humanloop.evaluations.create(
    config_id="string_example",
    evaluator_ids=["string_example"],
    dataset_id="string_example",
    project_id="project_id_example",
    provider_api_keys={},
)
```

#### ⚙️ Parameters

##### config_id: `str`

ID of the config to evaluate. Starts with `config_`.

##### evaluator_ids: [`CreateEvaluationRequestEvaluatorIds`](./humanloop/type/create_evaluation_request_evaluator_ids.py)

##### dataset_id: `str`

ID of the dataset to use in this evaluation. Starts with `evts_`.

##### project_id: `str`

String ID of project. Starts with `pr_`.

##### provider_api_keys: [`ProviderApiKeys`](./humanloop/type/provider_api_keys.py)


API keys required by each provider to make API calls. The API keys provided here are not stored by Humanloop. If not specified here, Humanloop will fall back to the key saved to your organization. Ensure you provide an API key for the provider for the model config you are evaluating, or have one saved to your organization.

#### ⚙️ Request Body

[`CreateEvaluationRequest`](./humanloop/type/create_evaluation_request.py)
#### 🔄 Return

[EvaluationResponse](./humanloop/type/evaluation_response.py)

#### 🌐 Endpoint

`/projects/{project_id}/evaluations` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.evaluations.get`

Get evaluation by ID.

#### 🛠️ Usage

```python
get_response = humanloop.evaluations.get(
    id="id_example",
    evaluator_aggregates=True,
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of evaluation run. Starts with `ev_`.

##### evaluator_aggregates: `bool`

Whether to include evaluator aggregates in the response.

#### 🔄 Return

[EvaluationResponse](./humanloop/type/evaluation_response.py)

#### 🌐 Endpoint

`/evaluations/{id}` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.evaluations.list_all_for_project`

Get all the evaluations associated with your project.

#### 🛠️ Usage

```python
list_all_for_project_response = humanloop.evaluations.list_all_for_project(
    project_id="project_id_example",
    evaluator_aggregates=True,
)
```

#### ⚙️ Parameters

##### project_id: `str`

String ID of project. Starts with `pr_`.

##### evaluator_aggregates: `bool`

Whether to include evaluator aggregates in the response.

#### 🔄 Return

[EvaluationsGetForProjectResponse](./humanloop/type/evaluations_get_for_project_response.py)

#### 🌐 Endpoint

`/projects/{project_id}/evaluations` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.evaluations.list_datapoints`

Get testcases by evaluation ID.

#### 🛠️ Usage

```python
list_datapoints_response = humanloop.evaluations.list_datapoints(
    id="id_example",
    page=1,
    size=10,
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of evaluation. Starts with `ev_`.

##### page: `int`

Page to fetch. Starts from 1.

##### size: `int`

Number of evaluation results to retrieve.

#### 🌐 Endpoint

`/evaluations/{id}/datapoints` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.evaluators.create`

Create an evaluator within your organization.

#### 🛠️ Usage

```python
create_response = humanloop.evaluators.create(
    description="string_example",
    name="string_example",
    code="string_example",
    arguments_type="string_example",
    return_type="string_example",
)
```

#### ⚙️ Parameters

##### description: `str`

The description of the evaluator.

##### name: `str`

The name of the evaluator.

##### code: `str`

The code for the evaluator. This code will be executed in a sandboxed environment.

##### arguments_type: [`EvaluatorArgumentsType`](./humanloop/type/evaluator_arguments_type.py)

Whether this evaluator is target-free or target-required.

##### return_type: [`EvaluatorReturnTypeEnum`](./humanloop/type/evaluator_return_type_enum.py)

The type of the return value of the evaluator.

#### ⚙️ Request Body

[`CreateEvaluatorRequest`](./humanloop/type/create_evaluator_request.py)
#### 🔄 Return

[EvaluatorResponse](./humanloop/type/evaluator_response.py)

#### 🌐 Endpoint

`/evaluators` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.evaluators.delete`

Delete an evaluator within your organization.

#### 🛠️ Usage

```python
humanloop.evaluators.delete(
    id="id_example",
)
```

#### ⚙️ Parameters

##### id: `str`

#### 🌐 Endpoint

`/evaluators/{id}` `delete`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.evaluators.list`

Get all evaluators within your organization.

#### 🛠️ Usage

```python
list_response = humanloop.evaluators.list()
```

#### 🔄 Return

[EvaluatorsListResponse](./humanloop/type/evaluators_list_response.py)

#### 🌐 Endpoint

`/evaluators` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.evaluators.update`

Update an evaluator within your organization.

#### 🛠️ Usage

```python
update_response = humanloop.evaluators.update(
    id="id_example",
    description="string_example",
    name="string_example",
    code="string_example",
    arguments_type="string_example",
    return_type="string_example",
)
```

#### ⚙️ Parameters

##### id: `str`

##### description: `str`

The description of the evaluator.

##### name: `str`

The name of the evaluator.

##### code: `str`

The code for the evaluator. This code will be executed in a sandboxed environment.

##### arguments_type: [`EvaluatorArgumentsType`](./humanloop/type/evaluator_arguments_type.py)

Whether this evaluator is target-free or target-required.

##### return_type: [`EvaluatorReturnTypeEnum`](./humanloop/type/evaluator_return_type_enum.py)

The type of the return value of the evaluator.

#### ⚙️ Request Body

[`UpdateEvaluatorRequest`](./humanloop/type/update_evaluator_request.py)
#### 🔄 Return

[EvaluatorResponse](./humanloop/type/evaluator_response.py)

#### 🌐 Endpoint

`/evaluators/{id}` `patch`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.experiments.create`

Create an experiment for your project.  You can optionally specify IDs of your project's model configs to include in the experiment, along with a set of labels to consider as positive feedback and whether the experiment should be set as active.

#### 🛠️ Usage

```python
create_response = humanloop.experiments.create(
    name="string_example",
    positive_labels=[
        {
            "type": "type_example",
            "value": "value_example",
        }
    ],
    project_id="project_id_example",
    config_ids=["string_example"],
    set_active=False,
)
```

#### ⚙️ Parameters

##### name: `str`

Name of experiment.

##### positive_labels: List[[`PositiveLabel`](./humanloop/type/positive_label.py)]

Feedback labels to treat as positive user feedback. Used to monitor the performance of model configs in the experiment.

##### project_id: `str`

String ID of project. Starts with `pr_`.

##### config_ids: [`CreateExperimentRequestConfigIds`](./humanloop/type/create_experiment_request_config_ids.py)

##### set_active: `bool`

Whether to set the created project as the project's active experiment.

#### ⚙️ Request Body

[`CreateExperimentRequest`](./humanloop/type/create_experiment_request.py)
#### 🔄 Return

[ExperimentResponse](./humanloop/type/experiment_response.py)

#### 🌐 Endpoint

`/projects/{project_id}/experiments` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.experiments.delete`

Delete the experiment with the specified ID.

#### 🛠️ Usage

```python
humanloop.experiments.delete(
    experiment_id="experiment_id_example",
)
```

#### ⚙️ Parameters

##### experiment_id: `str`

String ID of experiment. Starts with `exp_`.

#### 🌐 Endpoint

`/experiments/{experiment_id}` `delete`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.experiments.list`

Get an array of experiments associated to your project.

#### 🛠️ Usage

```python
list_response = humanloop.experiments.list(
    project_id="project_id_example",
)
```

#### ⚙️ Parameters

##### project_id: `str`

String ID of project. Starts with `pr_`.

#### 🔄 Return

[ExperimentsListResponse](./humanloop/type/experiments_list_response.py)

#### 🌐 Endpoint

`/projects/{project_id}/experiments` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.experiments.sample`

Samples a model config from the experiment's active model configs.

#### 🛠️ Usage

```python
sample_response = humanloop.experiments.sample(
    experiment_id="experiment_id_example",
)
```

#### ⚙️ Parameters

##### experiment_id: `str`

String ID of experiment. Starts with `exp_`.

#### 🔄 Return

[GetModelConfigResponse](./humanloop/type/get_model_config_response.py)

#### 🌐 Endpoint

`/experiments/{experiment_id}/model-config` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.experiments.update`

Update your experiment, including registering and de-registering model configs.

#### 🛠️ Usage

```python
update_response = humanloop.experiments.update(
    experiment_id="experiment_id_example",
    name="string_example",
    positive_labels=[
        {
            "type": "type_example",
            "value": "value_example",
        }
    ],
    config_ids_to_register=["string_example"],
    config_ids_to_deregister=["string_example"],
)
```

#### ⚙️ Parameters

##### experiment_id: `str`

String ID of experiment. Starts with `exp_`.

##### name: `str`

Name of experiment.

##### positive_labels: List[[`PositiveLabel`](./humanloop/type/positive_label.py)]

Feedback labels to treat as positive user feedback. Used to monitor the performance of model configs in the experiment.

##### config_ids_to_register: [`UpdateExperimentRequestConfigIdsToRegister`](./humanloop/type/update_experiment_request_config_ids_to_register.py)

##### config_ids_to_deregister: [`UpdateExperimentRequestConfigIdsToDeregister`](./humanloop/type/update_experiment_request_config_ids_to_deregister.py)

#### ⚙️ Request Body

[`UpdateExperimentRequest`](./humanloop/type/update_experiment_request.py)
#### 🔄 Return

[ExperimentResponse](./humanloop/type/experiment_response.py)

#### 🌐 Endpoint

`/experiments/{experiment_id}` `patch`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.feedback`

Submit an array of feedback for existing `data_ids`

#### 🛠️ Usage

```python
feedback_response = humanloop.feedback(
    body=[
        {
            "type": "string_example",
        }
    ],
    type="string_example",
    value="string_example",
    data_id="string_example",
    user="string_example",
    created_at="1970-01-01T00:00:00.00Z",
    unset=True,
)
```

#### ⚙️ Parameters

##### type: Union[[`FeedbackType`](./humanloop/type/feedback_type.py), `str`]


The type of feedback. The default feedback types available are 'rating', 'action', 'issue', 'correction', and 'comment'.

##### value: `str`

The feedback value to be set. This field should be left blank when unsetting 'rating', 'correction' or 'comment', but is required otherwise.

##### data_id: `str`

ID to associate the feedback to a previously logged datapoint.

##### user: `str`

A unique identifier to who provided the feedback.

##### created_at: `datetime`

User defined timestamp for when the feedback was created. 

##### unset: `bool`

If true, the value for this feedback is unset.

#### ⚙️ Request Body

[`FeedbackSubmitRequest`](./humanloop/type/feedback_submit_request.py)
#### 🔄 Return

[FeedbackSubmitResponse](./humanloop/type/feedback_submit_response.py)

#### 🌐 Endpoint

`/feedback` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.finetunes.create`

Trigger the fine-tuning process for a specific base model and data snapshot.

#### 🛠️ Usage

```python
create_response = humanloop.finetunes.create(
    name="string_example",
    dataset_id="string_example",
    config={
        "base_model": "base_model_example",
    },
    project_id="project_id_example",
    metadata={},
    provider_api_keys={},
)
```

#### ⚙️ Parameters

##### name: `str`

User defined friendly name for a finetuning run

##### dataset_id: `str`

ID of dataset used for finetuning

##### config: [`FinetuneConfig`](./humanloop/type/finetune_config.py)


Configuration and hyper-parameters for the fine-tuning process

##### project_id: `str`

##### metadata: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

Any additional metadata that you would like to log for reference.

##### provider_api_keys: [`ProviderApiKeys`](./humanloop/type/provider_api_keys.py)


API keys required by each provider to make API calls. The API keys provided here are not stored by Humanloop. If not specified here, Humanloop will fall back to the key saved to your organization.

#### ⚙️ Request Body

[`FinetuneRequest`](./humanloop/type/finetune_request.py)
#### 🔄 Return

[FinetuneResponse](./humanloop/type/finetune_response.py)

#### 🌐 Endpoint

`/projects/{project_id}/finetunes` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.finetunes.list_all_for_project`

Get a list of all fine-tuned models associated to a given project.

#### 🛠️ Usage

```python
list_all_for_project_response = humanloop.finetunes.list_all_for_project(
    project_id="project_id_example",
)
```

#### ⚙️ Parameters

##### project_id: `str`

#### 🔄 Return

[FinetunesListAllForProjectResponse](./humanloop/type/finetunes_list_all_for_project_response.py)

#### 🌐 Endpoint

`/projects/{project_id}/finetunes` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.finetunes.summary`

Checks data for errors and generates finetune data summary.  Does not actually trigger the finetuning process or persist the data.

#### 🛠️ Usage

```python
summary_response = humanloop.finetunes.summary(
    name="string_example",
    dataset_id="string_example",
    config={
        "base_model": "base_model_example",
    },
    project_id="project_id_example",
    metadata={},
    provider_api_keys={},
)
```

#### ⚙️ Parameters

##### name: `str`

User defined friendly name for a finetuning run

##### dataset_id: `str`

ID of dataset used for finetuning

##### config: [`FinetuneConfig`](./humanloop/type/finetune_config.py)


Configuration and hyper-parameters for the fine-tuning process

##### project_id: `str`

##### metadata: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

Any additional metadata that you would like to log for reference.

##### provider_api_keys: [`ProviderApiKeys`](./humanloop/type/provider_api_keys.py)


API keys required by each provider to make API calls. The API keys provided here are not stored by Humanloop. If not specified here, Humanloop will fall back to the key saved to your organization.

#### ⚙️ Request Body

[`FinetuneRequest`](./humanloop/type/finetune_request.py)
#### 🔄 Return

[FinetuneDataSummary](./humanloop/type/finetune_data_summary.py)

#### 🌐 Endpoint

`/projects/{project_id}/finetunes/summary` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.finetunes.update`

Update user-specified attributes of the specified finetuned models.

#### 🛠️ Usage

```python
update_response = humanloop.finetunes.update(
    id="id_example",
    project_id="project_id_example",
    name="string_example",
)
```

#### ⚙️ Parameters

##### id: `str`

##### project_id: `str`

##### name: `str`

#### ⚙️ Request Body

[`UpdateFinetuneRequest`](./humanloop/type/update_finetune_request.py)
#### 🔄 Return

[FinetuneResponse](./humanloop/type/finetune_response.py)

#### 🌐 Endpoint

`/finetunes/{id}` `patch`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.log`

Log a datapoint or array of datapoints to your Humanloop project.

#### 🛠️ Usage

```python
log_response = humanloop.log(
    body=[{}],
    project="string_example",
    project_id="string_example",
    session_id="string_example",
    session_reference_id="string_example",
    parent_id="string_example",
    parent_reference_id="string_example",
    inputs={},
    source="string_example",
    metadata={},
    reference_id="string_example",
    trial_id="string_example",
    messages=[
        {
            "role": "user",
        }
    ],
    output="string_example",
    config={
        "type": "AgentConfigRequest",
        "agent_class": "agent_class_example",
        "model_config": {
            "model": "model_example",
            "max_tokens": -1,
            "temperature": 1,
            "top_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "type": "model",
        },
    },
    feedback={
        "type": "string_example",
        "value": 3.14,
    },
    created_at="1970-01-01T00:00:00.00Z",
    error="string_example",
    duration=3.14,
)
```

#### ⚙️ Parameters

##### project: `str`

Unique project name. If no project exists with this name, a new project will be created.

##### project_id: `str`

Unique ID of a project to associate to the log. Either this or `project` must be provided.

##### session_id: `str`

ID of the session to associate the datapoint.

##### session_reference_id: `str`

A unique string identifying the session to associate the datapoint to. Allows you to log multiple datapoints to a session (using an ID kept by your internal systems) by passing the same `session_reference_id` in subsequent log requests. Specify at most one of this or `session_id`.

##### parent_id: `str`

ID associated to the parent datapoint in a session.

##### parent_reference_id: `str`

A unique string identifying the previously-logged parent datapoint in a session. Allows you to log nested datapoints with your internal system IDs by passing the same reference ID as `parent_id` in a prior log request. Specify at most one of this or `parent_id`. Note that this cannot refer to a datapoint being logged in the same request.

##### inputs: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

The inputs passed to the prompt template.

##### source: `str`

Identifies where the model was called from.

##### metadata: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

Any additional metadata to record.

##### reference_id: `str`

A unique string to reference the datapoint. Allows you to log nested datapoints with your internal system IDs by passing the same reference ID as `parent_id` in a subsequent log request.

##### trial_id: `str`

Unique ID of an experiment trial to associate to the log.

##### messages: List[[`ChatMessage`](./humanloop/type/chat_message.py)]

The messages passed to the to provider chat endpoint.

##### output: `str`

Generated output from your model for the provided inputs. Can be `None` if logging an error, or if logging a parent datapoint with the intention to populate it later

##### config: Union[[`ModelConfigRequest`](./humanloop/type/model_config_request.py), [`ToolConfigRequest`](./humanloop/type/tool_config_request.py), [`GenericConfigRequest`](./humanloop/type/generic_config_request.py), [`AgentConfigRequest`](./humanloop/type/agent_config_request.py)]


The model config used for this generation. Required unless `trial_id` is provided.

##### feedback: Union[[`Feedback`](./humanloop/type/feedback.py), List[[`Feedback`](./humanloop/type/feedback.py)]]


Optional parameter to provide feedback with your logged datapoint.

##### created_at: `datetime`

User defined timestamp for when the log was created. 

##### error: `str`

Error message if the log is an error.

##### duration: `Union[int, float]`

Duration of the logged event in seconds.

#### ⚙️ Request Body

[`LogDatapointRequest`](./humanloop/type/log_datapoint_request.py)
#### 🔄 Return

[LogsLogResponse](./humanloop/type/logs_log_response.py)

#### 🌐 Endpoint

`/logs` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.logs.update`

Update a logged datapoint in your Humanloop project.

#### 🛠️ Usage

```python
update_response = humanloop.logs.update(
    id="id_example",
    output="string_example",
    error="string_example",
    duration=3.14,
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of logged datapoint to return. Starts with `data_`.

##### output: `str`

Generated output from your model for the provided inputs.

##### error: `str`

Error message if the log is an error.

##### duration: `Union[int, float]`

Duration of the logged event in seconds.

#### ⚙️ Request Body

[`UpdateLogRequest`](./humanloop/type/update_log_request.py)
#### 🔄 Return

[LogResponse](./humanloop/type/log_response.py)

#### 🌐 Endpoint

`/logs/{id}` `patch`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.logs.update_by_ref`

Update a logged datapoint by its reference ID.  The `reference_id` query parameter must be provided, and refers to the `reference_id` of a previously-logged datapoint.

#### 🛠️ Usage

```python
update_by_ref_response = humanloop.logs.update_by_ref(
    reference_id="reference_id_example",
    output="string_example",
    error="string_example",
    duration=3.14,
)
```

#### ⚙️ Parameters

##### reference_id: `str`

A unique string to reference the datapoint. Identifies the logged datapoint created with the same `reference_id`.

##### output: `str`

Generated output from your model for the provided inputs.

##### error: `str`

Error message if the log is an error.

##### duration: `Union[int, float]`

Duration of the logged event in seconds.

#### ⚙️ Request Body

[`UpdateLogRequest`](./humanloop/type/update_log_request.py)
#### 🔄 Return

[LogResponse](./humanloop/type/log_response.py)

#### 🌐 Endpoint

`/logs` `patch`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.model_configs.get`

Get a specific model config by ID.

#### 🛠️ Usage

```python
get_response = humanloop.model_configs.get(
    id="id_example",
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of the model config. Starts with `config_`.

#### 🔄 Return

[ModelConfigResponse](./humanloop/type/model_config_response.py)

#### 🌐 Endpoint

`/model-configs/{id}` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.model_configs.register`

Register a model config to a project and optionally add it to an experiment.  If the project name provided does not exist, a new project will be created automatically.  If an experiment name is provided, the specified experiment must already exist. Otherwise, an error will be raised.  If the model config is the first to be associated to the project, it will be set as the active model config.

#### 🛠️ Usage

```python
register_response = humanloop.model_configs.register(
    model="string_example",
    description="string_example",
    name="string_example",
    provider="string_example",
    max_tokens=-1,
    temperature=1,
    top_p=1,
    stop="string_example",
    presence_penalty=0,
    frequency_penalty=0,
    other={},
    project="string_example",
    project_id="string_example",
    experiment="string_example",
    prompt_template="string_example",
    chat_template=[
        {
            "role": "user",
        }
    ],
    endpoint="string_example",
    tools=[
        {
            "name": "name_example",
        }
    ],
)
```

#### ⚙️ Parameters

##### model: `str`

The model instance used. E.g. text-davinci-002.

##### description: `str`

A description of the model config.

##### name: `str`

A friendly display name for the model config. If not provided, a name will be generated.

##### provider: [`ModelProviders`](./humanloop/type/model_providers.py)

The company providing the underlying model service.

##### max_tokens: `int`

The maximum number of tokens to generate. Provide max_tokens=-1 to dynamically calculate the maximum number of tokens to generate given the length of the prompt

##### temperature: `Union[int, float]`

What sampling temperature to use when making a generation. Higher values means the model will be more creative.

##### top_p: `Union[int, float]`

An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.

##### stop: Union[`str`, `List[str]`]


The string (or list of strings) after which the model will stop generating. The returned text will not contain the stop sequence.

##### presence_penalty: `Union[int, float]`

Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the generation so far.

##### frequency_penalty: `Union[int, float]`

Number between -2.0 and 2.0. Positive values penalize new tokens based on how frequently they appear in the generation so far.

##### other: `Dict[str, Union[bool, date, datetime, dict, float, int, list, str, None]]`

Other parameter values to be passed to the provider call.

##### project: `str`

Unique project name. If it does not exist, a new project will be created.

##### project_id: `str`

Unique project ID

##### experiment: `str`

If specified, the model config will be added to this experiment. Experiments are used for A/B testing and optimizing hyperparameters.

##### prompt_template: `str`

Prompt template that will take your specified inputs to form your final request to the provider model. NB: Input variables within the prompt template should be specified with syntax: {{INPUT_NAME}}.

##### chat_template: List[[`ChatMessage`](./humanloop/type/chat_message.py)]

Messages prepended to the list of messages sent to the provider. These messages that will take your specified inputs to form your final request to the provider model. NB: Input variables within the prompt template should be specified with syntax: {{INPUT_NAME}}.

##### endpoint: [`ModelEndpoints`](./humanloop/type/model_endpoints.py)

Which of the providers model endpoints to use. For example Complete or Edit.

##### tools: List[[`ModelConfigToolRequest`](./humanloop/type/model_config_tool_request.py)]

Make tools available to OpenAIs chat model as functions.

#### ⚙️ Request Body

[`ProjectModelConfigRequest`](./humanloop/type/project_model_config_request.py)
#### 🔄 Return

[ProjectConfigResponse](./humanloop/type/project_config_response.py)

#### 🌐 Endpoint

`/model-configs` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.projects.create`

Create a new project.

#### 🛠️ Usage

```python
create_response = humanloop.projects.create(
    name="string_example",
    feedback_types=[
        {
            "type": "type_example",
        }
    ],
)
```

#### ⚙️ Parameters

##### name: `str`

Unique project name.

##### feedback_types: List[[`FeedbackTypeRequest`](./humanloop/type/feedback_type_request.py)]

Feedback types to be created.

#### ⚙️ Request Body

[`CreateProjectRequest`](./humanloop/type/create_project_request.py)
#### 🔄 Return

[ProjectResponse](./humanloop/type/project_response.py)

#### 🌐 Endpoint

`/projects` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.projects.create_feedback_type`

Create Feedback Type

#### 🛠️ Usage

```python
create_feedback_type_response = humanloop.projects.create_feedback_type(
    type="string_example",
    id="id_example",
    values=[
        {
            "value": "value_example",
            "sentiment": "positive",
        }
    ],
    _class="string_example",
)
```

#### ⚙️ Parameters

##### type: `str`

The type of feedback to update.

##### id: `str`

String ID of project. Starts with `pr_`.

##### values: List[[`FeedbackLabelRequest`](./humanloop/type/feedback_label_request.py)]

The feedback values to be available. This field should only be populated when updating a 'select' or 'multi_select' feedback class.

##### _class: [`FeedbackClass`](./humanloop/type/feedback_class.py)

The data type associated to this feedback type; whether it is a 'text'/'select'/'multi_select'. This is optional when updating the default feedback types (i.e. when `type` is 'rating', 'action' or 'issue').

#### ⚙️ Request Body

[`FeedbackTypeRequest`](./humanloop/type/feedback_type_request.py)
#### 🔄 Return

[FeedbackTypeModel](./humanloop/type/feedback_type_model.py)

#### 🌐 Endpoint

`/projects/{id}/feedback-types` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.projects.deactivate_config`

Remove the project's active config, if set.  This has no effect if the project does not have an active model config set.

#### 🛠️ Usage

```python
deactivate_config_response = humanloop.projects.deactivate_config(
    id="id_example",
    environment="string_example",
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of project. Starts with `pr_`.

##### environment: `str`

Name for the environment. E.g. 'producton'. If not provided, will delete the active config for the default environment.

#### 🔄 Return

[ProjectResponse](./humanloop/type/project_response.py)

#### 🌐 Endpoint

`/projects/{id}/active-config` `delete`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.projects.deactivate_experiment`

Remove the project's active experiment, if set.  This has no effect if the project does not have an active experiment set.

#### 🛠️ Usage

```python
deactivate_experiment_response = humanloop.projects.deactivate_experiment(
    id="id_example",
    environment="string_example",
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of project. Starts with `pr_`.

##### environment: `str`

Name for the environment. E.g. 'producton'. If not provided, will return the experiment for the default environment.

#### 🔄 Return

[ProjectResponse](./humanloop/type/project_response.py)

#### 🌐 Endpoint

`/projects/{id}/active-experiment` `delete`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.projects.delete_deployed_config`

Remove the model config deployed to environment.  This has no effect if the project does not have an active model config set.

#### 🛠️ Usage

```python
delete_deployed_config_response = humanloop.projects.delete_deployed_config(
    project_id="project_id_example",
    environment_id="environment_id_example",
)
```

#### ⚙️ Parameters

##### project_id: `str`

##### environment_id: `str`

#### 🌐 Endpoint

`/projects/{project_id}/deployed-config/{environment_id}` `delete`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.projects.deploy_config`

Deploy a model config to an environment.  If the environment already has a model config deployed, it will be replaced.

#### 🛠️ Usage

```python
deploy_config_response = humanloop.projects.deploy_config(
    project_id="project_id_example",
    config_id="string_example",
    experiment_id="string_example",
    environments=[
        {
            "id": "id_example",
        }
    ],
)
```

#### ⚙️ Parameters

##### project_id: `str`

##### config_id: `str`

Model config unique identifier generated by Humanloop.

##### experiment_id: `str`

String ID of experiment. Starts with `exp_`.

##### environments: List[[`EnvironmentRequest`](./humanloop/type/environment_request.py)]

List of environments to associate with the model config.

#### ⚙️ Request Body

[`EnvironmentProjectConfigRequest`](./humanloop/type/environment_project_config_request.py)
#### 🔄 Return

[ProjectsDeployConfigToEnvironmentsResponse](./humanloop/type/projects_deploy_config_to_environments_response.py)

#### 🌐 Endpoint

`/projects/{project_id}/deploy-config` `patch`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.projects.export`

Export all logged datapoints associated to your project.  Results are paginated and sorts the datapoints based on `created_at` in descending order.

#### 🛠️ Usage

```python
export_response = humanloop.projects.export(
    id="id_example",
    page=0,
    size=10,
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of project. Starts with `pr_`.

##### page: `int`

Page offset for pagination.

##### size: `int`

Page size for pagination. Number of logs to export.

#### 🌐 Endpoint

`/projects/{id}/export` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.projects.get`

Get a specific project.

#### 🛠️ Usage

```python
get_response = humanloop.projects.get(
    id="id_example",
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of project. Starts with `pr_`.

#### 🔄 Return

[ProjectResponse](./humanloop/type/project_response.py)

#### 🌐 Endpoint

`/projects/{id}` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.projects.get_active_config`

Retrieves a config to use to execute your model.  A config will be selected based on the project's active config/experiment settings.

#### 🛠️ Usage

```python
get_active_config_response = humanloop.projects.get_active_config(
    id="id_example",
    environment="string_example",
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of project. Starts with `pr_`.

##### environment: `str`

Name for the environment. E.g. 'producton'. If not provided, will return the active config for the default environment.

#### 🔄 Return

[GetModelConfigResponse](./humanloop/type/get_model_config_response.py)

#### 🌐 Endpoint

`/projects/{id}/active-config` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.projects.list`

Get a paginated list of projects.

#### 🛠️ Usage

```python
list_response = humanloop.projects.list(
    page=0,
    size=10,
    organization_id="string_example",
    filter="string_example",
    user_filter="string_example",
    sort_by="string_example",
    order="string_example",
)
```

#### ⚙️ Parameters

##### page: `int`

Page offset for pagination.

##### size: `int`

Page size for pagination. Number of projects to fetch.

##### organization_id: `str`

ID of organization that fetched projects belong to. Starts with `org_`.

##### filter: `str`

Case-insensitive filter for project name.

##### user_filter: `str`

Case-insensitive filter for users in the project. This filter matches against both email address and name of users.

##### sort_by: [`ProjectSortBy`](./humanloop/type/project_sort_by.py)

Field to sort projects by

##### order: [`SortOrder`](./humanloop/type/sort_order.py)

Direction to sort by.

#### 🌐 Endpoint

`/projects` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.projects.list_configs`

Get an array of configs associated to your project.

#### 🛠️ Usage

```python
list_configs_response = humanloop.projects.list_configs(
    id="id_example",
    evaluation_aggregates=True,
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of project. Starts with `pr_`.

##### evaluation_aggregates: `bool`

#### 🔄 Return

[ProjectsGetConfigsResponse](./humanloop/type/projects_get_configs_response.py)

#### 🌐 Endpoint

`/projects/{id}/configs` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.projects.list_deployed_configs`

Get an array of environments with the deployed configs associated to your project.

#### 🛠️ Usage

```python
list_deployed_configs_response = humanloop.projects.list_deployed_configs(
    id="id_example",
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of project. Starts with `pr_`.

#### 🔄 Return

[ProjectsGetDeployedConfigsResponse](./humanloop/type/projects_get_deployed_configs_response.py)

#### 🌐 Endpoint

`/projects/{id}/deployed-configs` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.projects.update`

Update a specific project.  Set the project's active model config/experiment by passing either `active_experiment_id` or `active_model_config_id`. These will be set to the Default environment unless a list of environments are also passed in specifically detailing which environments to assign the active config or experiment.  Set the feedback labels to be treated as positive user feedback used in calculating top-level project metrics by passing a list of labels in `positive_labels`.

#### 🛠️ Usage

```python
update_response = humanloop.projects.update(
    id="id_example",
    name="string_example",
    active_experiment_id="string_example",
    active_config_id="string_example",
    positive_labels=[
        {
            "type": "type_example",
            "value": "value_example",
        }
    ],
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of project. Starts with `pr_`.

##### name: `str`

The new unique project name. Caution, if you are using the project name as the unique identifier in your API calls, changing the name will break the calls.

##### active_experiment_id: `str`

ID for an experiment to set as the project's active deployment. Starts with 'exp_'. At most one of 'active_experiment_id' and 'active_model_config_id' can be set.

##### active_config_id: `str`

ID for a config to set as the project's active deployment. Starts with 'config_'. At most one of 'active_experiment_id' and 'active_config_id' can be set.

##### positive_labels: List[[`PositiveLabel`](./humanloop/type/positive_label.py)]

The full list of labels to treat as positive user feedback.

#### ⚙️ Request Body

[`UpdateProjectRequest`](./humanloop/type/update_project_request.py)
#### 🔄 Return

[ProjectResponse](./humanloop/type/project_response.py)

#### 🌐 Endpoint

`/projects/{id}` `patch`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.projects.update_feedback_types`

Update feedback types.  Allows enabling the available feedback types and setting status of feedback types/categorical values.  This behaves like an upsert; any feedback categorical values that do not already exist in the project will be created.

#### 🛠️ Usage

```python
update_feedback_types_response = humanloop.projects.update_feedback_types(
    body=[
        {
            "type": "type_example",
        }
    ],
    id="id_example",
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of project. Starts with `pr_`.

##### requestBody: [`ProjectsUpdateFeedbackTypesRequest`](./humanloop/type/projects_update_feedback_types_request.py)

#### 🔄 Return

[FeedbackTypes](./humanloop/type/feedback_types.py)

#### 🌐 Endpoint

`/projects/{id}/feedback-types` `patch`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.sessions.create`

Create a new session.  Returns a session ID that can be used to log datapoints to the session.

#### 🛠️ Usage

```python
create_response = humanloop.sessions.create()
```

#### 🔄 Return

[CreateSessionResponse](./humanloop/type/create_session_response.py)

#### 🌐 Endpoint

`/sessions` `post`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.sessions.get`

Get a session by ID.

#### 🛠️ Usage

```python
get_response = humanloop.sessions.get(
    id="id_example",
)
```

#### ⚙️ Parameters

##### id: `str`

String ID of session to return. Starts with `sesh_`.

#### 🔄 Return

[SessionResponse](./humanloop/type/session_response.py)

#### 🌐 Endpoint

`/sessions/{id}` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---

### `humanloop.sessions.list`

Get a page of sessions.

#### 🛠️ Usage

```python
list_response = humanloop.sessions.list(
    project_id="project_id_example",
    page=1,
    size=10,
)
```

#### ⚙️ Parameters

##### project_id: `str`

String ID of project to return sessions for. Sessions that contain any datapoints associated to this project will be returned. Starts with `pr_`.

##### page: `int`

Page to fetch. Starts from 1.

##### size: `int`

Number of sessions to retrieve.

#### 🌐 Endpoint

`/sessions` `get`

[🔙 **Back to Table of Contents**](#table-of-contents)

---


## Author
This Python package is automatically generated by [Konfig](https://konfigthis.com)
