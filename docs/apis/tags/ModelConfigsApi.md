# humanloop.model_configs

All URIs are relative to *https://api.humanloop.com/v4*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get**](#get) | **get** /model-configs/{id} | Get
[**register**](#register) | **post** /model-configs | Register

# **get**

Get a specific model config by ID.

### Example

```python
from pprint import pprint
from humanloop import Humanloop, ApiException

humanloop = Humanloop(
    openai_api_key="OPENAI_API_KEY",
    openai_azure_api_key="OPENAI_AZURE_API_KEY",
    openai_azure_endpoint_api_key="OPENAI_AZURE_ENDPOINT_API_KEY",
    ai21_api_key="AI21_API_KEY",
    mock_api_key="MOCK_API_KEY",
    anthropic_api_key="ANTHROPIC_API_KEY",
    cohere_api_key="COHERE_API_KEY",
    api_key="YOUR_API_KEY",
)

try:
    # Get
    get_response = humanloop.model_configs.get(
        id="id_example",  # required
    )
    pprint(get_response.body)
    pprint(get_response.body["id"])
    pprint(get_response.body["type"])
    pprint(get_response.body["model"])
    pprint(get_response.body["description"])
    pprint(get_response.body["other"])
    pprint(get_response.body["name"])
    pprint(get_response.body["provider"])
    pprint(get_response.body["max_tokens"])
    pprint(get_response.body["temperature"])
    pprint(get_response.body["top_p"])
    pprint(get_response.body["stop"])
    pprint(get_response.body["presence_penalty"])
    pprint(get_response.body["frequency_penalty"])
    pprint(get_response.body["prompt_template"])
    pprint(get_response.body["chat_template"])
    pprint(get_response.body["tool_configs"])
    pprint(get_response.body["endpoint"])
    pprint(get_response.headers)
    pprint(get_response.status)
    pprint(get_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ModelConfigsApi.get: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **register**

Register a model config to a project and optionally add it to an experiment.  If the project name provided does not exist, a new project will be created automatically.  If an experiment name is provided, the specified experiment must already exist. Otherwise, an error will be raised.  If the model config is the first to be associated to the project, it will be set as the active model config.

### Example

```python
from pprint import pprint
from humanloop import Humanloop, ApiException

humanloop = Humanloop(
    openai_api_key="OPENAI_API_KEY",
    openai_azure_api_key="OPENAI_AZURE_API_KEY",
    openai_azure_endpoint_api_key="OPENAI_AZURE_ENDPOINT_API_KEY",
    ai21_api_key="AI21_API_KEY",
    mock_api_key="MOCK_API_KEY",
    anthropic_api_key="ANTHROPIC_API_KEY",
    cohere_api_key="COHERE_API_KEY",
    api_key="YOUR_API_KEY",
)

try:
    # Register
    register_response = humanloop.model_configs.register(
        model="string_example",  # required
        description="string_example",  # optional
        name="string_example",  # optional
        provider="string_example",  # optional
        max_tokens=-1,  # optional
        temperature=1,  # optional
        top_p=1,  # optional
        stop="string_example",  # optional
        presence_penalty=0,  # optional
        frequency_penalty=0,  # optional
        other={},  # optional
        project="string_example",  # optional
        project_id="string_example",  # optional
        experiment="string_example",  # optional
        prompt_template="string_example",  # optional
        chat_template=[
            {
                "role": "user",
            }
        ],  # optional
        endpoint="string_example",  # optional
        tools=[
            {
                "name": "name_example",
            }
        ],  # optional
    )
    pprint(register_response.body)
    pprint(register_response.body["project_id"])
    pprint(register_response.body["project_name"])
    pprint(register_response.body["created_at"])
    pprint(register_response.body["updated_at"])
    pprint(register_response.body["last_used"])
    pprint(register_response.body["config"])
    pprint(register_response.body["feedback_stats"])
    pprint(register_response.body["num_datapoints"])
    pprint(register_response.body["experiment_id"])
    pprint(register_response.body["evaluation_aggregates"])
    pprint(register_response.headers)
    pprint(register_response.status)
    pprint(register_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ModelConfigsApi.register: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

