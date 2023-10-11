# humanloop.experiments

All URIs are relative to *https://api.humanloop.com/v4*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create**](#create) | **post** /projects/{project_id}/experiments | Create
[**delete**](#delete) | **delete** /experiments/{experiment_id} | Delete
[**list**](#list) | **get** /projects/{project_id}/experiments | List
[**sample**](#sample) | **get** /experiments/{experiment_id}/model-config | Sample
[**update**](#update) | **patch** /experiments/{experiment_id} | Update

# **create**

Create an experiment for your project.  You can optionally specify IDs of your project's model configs to include in the experiment, along with a set of labels to consider as positive feedback and whether the experiment should be set as active.

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
    create_response = humanloop.experiments.create(
        name="string_example",  # required
        positive_labels=[
            {
                "type": "type_example",
                "value": "value_example",
            }
        ],  # required
        project_id="project_id_example",  # required
        config_ids=["string_example"],  # optional
        set_active=False,  # optional
    )
    pprint(create_response.body)
    pprint(create_response.body["id"])
    pprint(create_response.body["project_id"])
    pprint(create_response.body["name"])
    pprint(create_response.body["status"])
    pprint(create_response.body["metric"])
    pprint(create_response.body["positive_labels"])
    pprint(create_response.body["created_at"])
    pprint(create_response.body["updated_at"])
    pprint(create_response.body["configs"])
    pprint(create_response.headers)
    pprint(create_response.status)
    pprint(create_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ExperimentsApi.create: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **delete**

Delete the experiment with the specified ID.

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
    # Delete
    humanloop.experiments.delete(
        experiment_id="experiment_id_example",  # required
    )
except ApiException as e:
    print("Exception when calling ExperimentsApi.delete: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **list**

Get an array of experiments associated to your project.

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
    # List
    list_response = humanloop.experiments.list(
        project_id="project_id_example",  # required
    )
    pprint(list_response.body)
    pprint(list_response.headers)
    pprint(list_response.status)
    pprint(list_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ExperimentsApi.list: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **sample**

Samples a model config from the experiment's active model configs.

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
    # Sample
    sample_response = humanloop.experiments.sample(
        experiment_id="experiment_id_example",  # required
    )
    pprint(sample_response.body)
    pprint(sample_response.body["project_id"])
    pprint(sample_response.body["project_name"])
    pprint(sample_response.body["created_at"])
    pprint(sample_response.body["updated_at"])
    pprint(sample_response.body["last_used"])
    pprint(sample_response.body["config"])
    pprint(sample_response.body["feedback_stats"])
    pprint(sample_response.body["num_datapoints"])
    pprint(sample_response.body["experiment_id"])
    pprint(sample_response.body["evaluation_aggregates"])
    pprint(sample_response.body["trial_id"])
    pprint(sample_response.headers)
    pprint(sample_response.status)
    pprint(sample_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ExperimentsApi.sample: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **update**

Update your experiment, including registering and de-registering model configs.

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
    # Update
    update_response = humanloop.experiments.update(
        experiment_id="experiment_id_example",  # required
        name="string_example",  # optional
        positive_labels=[
            {
                "type": "type_example",
                "value": "value_example",
            }
        ],  # optional
        config_ids_to_register=["string_example"],  # optional
        config_ids_to_deregister=["string_example"],  # optional
    )
    pprint(update_response.body)
    pprint(update_response.body["id"])
    pprint(update_response.body["project_id"])
    pprint(update_response.body["name"])
    pprint(update_response.body["status"])
    pprint(update_response.body["metric"])
    pprint(update_response.body["positive_labels"])
    pprint(update_response.body["created_at"])
    pprint(update_response.body["updated_at"])
    pprint(update_response.body["configs"])
    pprint(update_response.headers)
    pprint(update_response.status)
    pprint(update_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ExperimentsApi.update: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

