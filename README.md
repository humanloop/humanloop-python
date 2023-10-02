# humanloop

[![PyPI](https://img.shields.io/badge/PyPI-v0.5.15-blue)](https://pypi.org/project/humanloop/0.5.15)



## Requirements

Python >=3.7

## Installing

```sh
pip install humanloop==0.5.15
```

## Getting Started

```python
from pprint import pprint
from humanloop import Humanloop, ApiException

humanloop = Humanloop(
    api_key="YOUR_API_KEY",
    openai_api_key="YOUR_OPENAI_API_KEY",
    ai21_api_key="YOUR_AI21_API_KEY",
    mock_api_key="YOUR_MOCK_API_KEY",
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
    ai21_api_key="YOUR_AI21_API_KEY",
    mock_api_key="YOUR_MOCK_API_KEY",
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
    ai21_api_key="YOUR_AI21_API_KEY",
    mock_api_key="YOUR_MOCK_API_KEY",
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


## Documentation for API Endpoints

All URIs are relative to *https://api.humanloop.com/v4*

Tag | Method | HTTP request | Description
------------ | ------------- | ------------- | -------------
*Chats* | [**create**](docs/apis/tags/ChatsApi.md#create) | **POST** /chat | Get a chat response by providing details of the model configuration in the request.
*Chats* | [**create_deployed**](docs/apis/tags/ChatsApi.md#create_deployed) | **POST** /chat-deployed | Get a chat response using the project&#x27;s active deployment.  The active deployment can be a specific model configuration or an experiment.
*Chats* | [**create_experiment**](docs/apis/tags/ChatsApi.md#create_experiment) | **POST** /chat-experiment | Get a chat response for a specific experiment.
*Chats* | [**create_model_config**](docs/apis/tags/ChatsApi.md#create_model_config) | **POST** /chat-model-config | Get chat response for a specific model configuration.
*Completions* | [**create**](docs/apis/tags/CompletionsApi.md#create) | **POST** /completion | Create a completion by providing details of the model configuration in the request.
*Completions* | [**create_deployed**](docs/apis/tags/CompletionsApi.md#create_deployed) | **POST** /completion-deployed | Create a completion using the project&#x27;s active deployment.  The active deployment can be a specific model configuration or an experiment.
*Completions* | [**create_experiment**](docs/apis/tags/CompletionsApi.md#create_experiment) | **POST** /completion-experiment | Create a completion for a specific experiment.
*Completions* | [**create_model_config**](docs/apis/tags/CompletionsApi.md#create_model_config) | **POST** /completion-model-config | Create a completion for a specific model configuration.
*Evaluations* | [**create**](docs/apis/tags/EvaluationsApi.md#create) | **POST** /projects/{project_id}/evaluations | Create an evaluation.
*Evaluations* | [**get**](docs/apis/tags/EvaluationsApi.md#get) | **GET** /evaluations/{id} | Get evaluation by ID.
*Evaluations* | [**list_all_for_project**](docs/apis/tags/EvaluationsApi.md#list_all_for_project) | **GET** /projects/{project_id}/evaluations | Get all the evaluations associated with your project.
*Evaluations* | [**list_testcases**](docs/apis/tags/EvaluationsApi.md#list_testcases) | **GET** /evaluations/{id}/testcases | Get testcases by evaluation ID.
*Evaluators* | [**create**](docs/apis/tags/EvaluatorsApi.md#create) | **POST** /evaluators | Create an evaluator within your organization.
*Evaluators* | [**delete**](docs/apis/tags/EvaluatorsApi.md#delete) | **DELETE** /evaluators/{id} | Delete an evaluator within your organization.
*Evaluators* | [**list**](docs/apis/tags/EvaluatorsApi.md#list) | **GET** /evaluators | Get all evaluators within your organization.
*Evaluators* | [**update**](docs/apis/tags/EvaluatorsApi.md#update) | **PATCH** /evaluators/{id} | Update an evaluator within your organization.
*Experiments* | [**create**](docs/apis/tags/ExperimentsApi.md#create) | **POST** /projects/{project_id}/experiments | Create an experiment for your project.  You can optionally specify IDs of your project&#x27;s model configs to include in the experiment, along with a set of labels to consider as positive feedback and whether the experiment should be set as active.
*Experiments* | [**delete**](docs/apis/tags/ExperimentsApi.md#delete) | **DELETE** /experiments/{experiment_id} | Delete the experiment with the specified ID.
*Experiments* | [**list**](docs/apis/tags/ExperimentsApi.md#list) | **GET** /projects/{project_id}/experiments | Get an array of experiments associated to your project.
*Experiments* | [**sample**](docs/apis/tags/ExperimentsApi.md#sample) | **GET** /experiments/{experiment_id}/model-config | Samples a model config from the experiment&#x27;s active model configs.
*Experiments* | [**update**](docs/apis/tags/ExperimentsApi.md#update) | **PATCH** /experiments/{experiment_id} | Update your experiment, including registering and de-registering model configs.
*Feedback* | [**feedback**](docs/apis/tags/FeedbackApi.md#feedback) | **POST** /feedback | Submit an array of feedback for existing &#x60;data_ids&#x60;
*Logs* | [**log**](docs/apis/tags/LogsApi.md#log) | **POST** /logs | Log a datapoint or array of datapoints to your Humanloop project.
*Logs* | [**update**](docs/apis/tags/LogsApi.md#update) | **PATCH** /logs/{id} | Update a logged datapoint in your Humanloop project.
*Logs* | [**update_by_ref**](docs/apis/tags/LogsApi.md#update_by_ref) | **PATCH** /logs | Update a logged datapoint by its reference ID.  The &#x60;reference_id&#x60; query parameter must be provided, and refers to the &#x60;reference_id&#x60; of a previously-logged datapoint.
*Model Configs* | [**get**](docs/apis/tags/ModelConfigsApi.md#get) | **GET** /model-configs/{id} | Get a specific model config by ID.
*Model Configs* | [**register**](docs/apis/tags/ModelConfigsApi.md#register) | **POST** /model-configs | Register a model config to a project and optionally add it to an experiment.  If the project name provided does not exist, a new project will be created automatically.  If an experiment name is provided, the specified experiment must already exist. Otherwise, an error will be raised.  If the model config is the first to be associated to the project, it will be set as the active model config.
*Projects* | [**create**](docs/apis/tags/ProjectsApi.md#create) | **POST** /projects | Create a new project.
*Projects* | [**create_feedback_type**](docs/apis/tags/ProjectsApi.md#create_feedback_type) | **POST** /projects/{id}/feedback-types | 
*Projects* | [**deactivate_config**](docs/apis/tags/ProjectsApi.md#deactivate_config) | **DELETE** /projects/{id}/active-config | Remove the project&#x27;s active config, if set.  This has no effect if the project does not have an active model config set.
*Projects* | [**deactivate_experiment**](docs/apis/tags/ProjectsApi.md#deactivate_experiment) | **DELETE** /projects/{id}/active-experiment | Remove the project&#x27;s active experiment, if set.  This has no effect if the project does not have an active experiment set.
*Projects* | [**delete_deployed_config**](docs/apis/tags/ProjectsApi.md#delete_deployed_config) | **DELETE** /projects/{project_id}/deployed-config/{environment_id} | Remove the model config deployed to environment.  This has no effect if the project does not have an active model config set.
*Projects* | [**deploy_config**](docs/apis/tags/ProjectsApi.md#deploy_config) | **PATCH** /projects/{project_id}/deploy-config | Deploy a model config to an environment.  If the environment already has a model config deployed, it will be replaced.
*Projects* | [**export**](docs/apis/tags/ProjectsApi.md#export) | **POST** /projects/{id}/export | Export all logged datapoints associated to your project.  Results are paginated and sorts the datapoints based on &#x60;created_at&#x60; in descending order.
*Projects* | [**get**](docs/apis/tags/ProjectsApi.md#get) | **GET** /projects/{id} | Get a specific project.
*Projects* | [**get_active_config**](docs/apis/tags/ProjectsApi.md#get_active_config) | **GET** /projects/{id}/active-config | Retrieves a config to use to execute your model.  A config will be selected based on the project&#x27;s active config/experiment settings.
*Projects* | [**list**](docs/apis/tags/ProjectsApi.md#list) | **GET** /projects | Get a paginated list of projects.
*Projects* | [**list_configs**](docs/apis/tags/ProjectsApi.md#list_configs) | **GET** /projects/{id}/configs | Get an array of configs associated to your project.
*Projects* | [**list_deployed_configs**](docs/apis/tags/ProjectsApi.md#list_deployed_configs) | **GET** /projects/{id}/deployed-configs | Get an array of environments with the deployed configs associated to your project.
*Projects* | [**update**](docs/apis/tags/ProjectsApi.md#update) | **PATCH** /projects/{id} | Update a specific project.  Set the project&#x27;s active model config/experiment by passing either &#x60;active_experiment_id&#x60; or &#x60;active_model_config_id&#x60;. These will be set to the Default environment unless a list of environments are also passed in specifically detailing which environments to assign the active config or experiment.  Set the feedback labels to be treated as positive user feedback used in calculating top-level project metrics by passing a list of labels in &#x60;positive_labels&#x60;.
*Projects* | [**update_feedback_types**](docs/apis/tags/ProjectsApi.md#update_feedback_types) | **PATCH** /projects/{id}/feedback-types | Update feedback types.  Allows enabling the available feedback types and setting status of feedback types/categorical values.  This behaves like an upsert; any feedback categorical values that do not already exist in the project will be created.
*Sessions* | [**create**](docs/apis/tags/SessionsApi.md#create) | **POST** /sessions | Create a new session.  Returns a session ID that can be used to log datapoints to the session.
*Sessions* | [**get**](docs/apis/tags/SessionsApi.md#get) | **GET** /sessions/{id} | Get a session by ID.
*Sessions* | [**list**](docs/apis/tags/SessionsApi.md#list) | **GET** /sessions | Get a page of sessions.
*Testcases* | [**delete**](docs/apis/tags/TestcasesApi.md#delete) | **DELETE** /testcases | Delete a list of testsets by their IDs.
*Testcases* | [**get**](docs/apis/tags/TestcasesApi.md#get) | **GET** /testcases/{id} | Get a testcase by ID.
*Testcases* | [**update**](docs/apis/tags/TestcasesApi.md#update) | **PATCH** /testcases/{id} | Edit the input, messages and criteria fields of a testcase. The fields passed in the request are the ones edited.  Passing &#x60;null&#x60; as a value for a field will delete that field. In order to signify not changing a field, it should be omitted from the request body.
*Testsets* | [**create**](docs/apis/tags/TestsetsApi.md#create) | **POST** /projects/{project_id}/testsets | Create a new testset for a project.
*Testsets* | [**create_testcase**](docs/apis/tags/TestsetsApi.md#create_testcase) | **POST** /testsets/{testset_id}/testcases | Create a new testcase for a testset.
*Testsets* | [**delete**](docs/apis/tags/TestsetsApi.md#delete) | **DELETE** /testsets/{id} | Delete a testset by ID.
*Testsets* | [**get**](docs/apis/tags/TestsetsApi.md#get) | **GET** /testsets/{id} | Get a single testset by ID.
*Testsets* | [**list_all_for_project**](docs/apis/tags/TestsetsApi.md#list_all_for_project) | **GET** /projects/{project_id}/testsets | Get all testsets for a project.
*Testsets* | [**list_testcases**](docs/apis/tags/TestsetsApi.md#list_testcases) | **GET** /testsets/{testset_id}/testcases | Get testcases for a testset.
*Testsets* | [**update**](docs/apis/tags/TestsetsApi.md#update) | **PATCH** /testsets/{id} | Update a testset by ID.


## Author
This Python package is automatically generated by [Konfig](https://konfigthis.com)
