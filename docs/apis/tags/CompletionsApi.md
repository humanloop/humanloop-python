# humanloop.completions

All URIs are relative to *https://api.humanloop.com/v4*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create**](#create) | **post** /completion | Create
[**create_deployed**](#create_deployed) | **post** /completion-deployed | Completion Deployed
[**create_experiment**](#create_experiment) | **post** /completion-experiment | Completion Experiment
[**create_model_config**](#create_model_config) | **post** /completion-model-config | Completion Model Config

# **create**

Create a completion by providing details of the model configuration in the request.

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
    # Create
    create_response = humanloop.complete(
        model_config={
            "model": "model_example",
            "max_tokens": -1,
            "temperature": 1,
            "top_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "prompt_template": "{{question}}",
        },  # required
        project="string_example",  # optional
        project_id="string_example",  # optional
        session_id="string_example",  # optional
        session_reference_id="string_example",  # optional
        parent_id="string_example",  # optional
        parent_reference_id="string_example",  # optional
        inputs={},  # optional
        source="string_example",  # optional
        metadata={},  # optional
        provider_api_keys={},  # optional
        num_samples=1,  # optional
        logprobs=1,  # optional
        stream=False,  # optional
        suffix="string_example",  # optional
        user="string_example",  # optional
    )
    pprint(create_response.body)
    pprint(create_response.body["data"])
    pprint(create_response.body["provider_responses"])
    pprint(create_response.body["project_id"])
    pprint(create_response.body["num_samples"])
    pprint(create_response.body["logprobs"])
    pprint(create_response.body["suffix"])
    pprint(create_response.body["user"])
    pprint(create_response.body["usage"])
    pprint(create_response.body["metadata"])
    pprint(create_response.headers)
    pprint(create_response.status)
    pprint(create_response.round_trip_time)
except ApiException as e:
    print("Exception when calling CompletionsApi.create: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_deployed**

Create a completion using the project's active deployment.  The active deployment can be a specific model configuration or an experiment.

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
    # Completion Deployed
    create_deployed_response = humanloop.complete_deployed(
        project="string_example",  # optional
        project_id="string_example",  # optional
        session_id="string_example",  # optional
        session_reference_id="string_example",  # optional
        parent_id="string_example",  # optional
        parent_reference_id="string_example",  # optional
        inputs={},  # optional
        source="string_example",  # optional
        metadata={},  # optional
        provider_api_keys={},  # optional
        num_samples=1,  # optional
        logprobs=1,  # optional
        stream=False,  # optional
        suffix="string_example",  # optional
        user="string_example",  # optional
        environment="string_example",  # optional
    )
    pprint(create_deployed_response.body)
    pprint(create_deployed_response.body["data"])
    pprint(create_deployed_response.body["provider_responses"])
    pprint(create_deployed_response.body["project_id"])
    pprint(create_deployed_response.body["num_samples"])
    pprint(create_deployed_response.body["logprobs"])
    pprint(create_deployed_response.body["suffix"])
    pprint(create_deployed_response.body["user"])
    pprint(create_deployed_response.body["usage"])
    pprint(create_deployed_response.body["metadata"])
    pprint(create_deployed_response.headers)
    pprint(create_deployed_response.status)
    pprint(create_deployed_response.round_trip_time)
except ApiException as e:
    print("Exception when calling CompletionsApi.create_deployed: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_experiment**

Create a completion for a specific experiment.

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
    # Completion Experiment
    create_experiment_response = humanloop.complete_experiment(
        experiment_id="string_example",  # required
        project="string_example",  # optional
        project_id="string_example",  # optional
        session_id="string_example",  # optional
        session_reference_id="string_example",  # optional
        parent_id="string_example",  # optional
        parent_reference_id="string_example",  # optional
        inputs={},  # optional
        source="string_example",  # optional
        metadata={},  # optional
        provider_api_keys={},  # optional
        num_samples=1,  # optional
        logprobs=1,  # optional
        stream=False,  # optional
        suffix="string_example",  # optional
        user="string_example",  # optional
    )
    pprint(create_experiment_response.body)
    pprint(create_experiment_response.body["data"])
    pprint(create_experiment_response.body["provider_responses"])
    pprint(create_experiment_response.body["project_id"])
    pprint(create_experiment_response.body["num_samples"])
    pprint(create_experiment_response.body["logprobs"])
    pprint(create_experiment_response.body["suffix"])
    pprint(create_experiment_response.body["user"])
    pprint(create_experiment_response.body["usage"])
    pprint(create_experiment_response.body["metadata"])
    pprint(create_experiment_response.headers)
    pprint(create_experiment_response.status)
    pprint(create_experiment_response.round_trip_time)
except ApiException as e:
    print("Exception when calling CompletionsApi.create_experiment: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_model_config**

Create a completion for a specific model configuration.

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
    # Completion Model Config
    create_model_config_response = humanloop.complete_model_configuration(
        model_config_id="string_example",  # required
        project="string_example",  # optional
        project_id="string_example",  # optional
        session_id="string_example",  # optional
        session_reference_id="string_example",  # optional
        parent_id="string_example",  # optional
        parent_reference_id="string_example",  # optional
        inputs={},  # optional
        source="string_example",  # optional
        metadata={},  # optional
        provider_api_keys={},  # optional
        num_samples=1,  # optional
        logprobs=1,  # optional
        stream=False,  # optional
        suffix="string_example",  # optional
        user="string_example",  # optional
    )
    pprint(create_model_config_response.body)
    pprint(create_model_config_response.body["data"])
    pprint(create_model_config_response.body["provider_responses"])
    pprint(create_model_config_response.body["project_id"])
    pprint(create_model_config_response.body["num_samples"])
    pprint(create_model_config_response.body["logprobs"])
    pprint(create_model_config_response.body["suffix"])
    pprint(create_model_config_response.body["user"])
    pprint(create_model_config_response.body["usage"])
    pprint(create_model_config_response.body["metadata"])
    pprint(create_model_config_response.headers)
    pprint(create_model_config_response.status)
    pprint(create_model_config_response.round_trip_time)
except ApiException as e:
    print("Exception when calling CompletionsApi.create_model_config: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

