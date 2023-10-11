# humanloop.projects

All URIs are relative to *https://api.humanloop.com/v4*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create**](#create) | **post** /projects | Create
[**create_feedback_type**](#create_feedback_type) | **post** /projects/{id}/feedback-types | Create Feedback Type
[**deactivate_config**](#deactivate_config) | **delete** /projects/{id}/active-config | Deactivate Config
[**deactivate_experiment**](#deactivate_experiment) | **delete** /projects/{id}/active-experiment | Deactivate Experiment
[**delete_deployed_config**](#delete_deployed_config) | **delete** /projects/{project_id}/deployed-config/{environment_id} | Delete Deployed Config
[**deploy_config**](#deploy_config) | **patch** /projects/{project_id}/deploy-config | Deploy Config
[**export**](#export) | **post** /projects/{id}/export | Export
[**get**](#get) | **get** /projects/{id} | Get
[**get_active_config**](#get_active_config) | **get** /projects/{id}/active-config | Get Active Config
[**list**](#list) | **get** /projects | List
[**list_configs**](#list_configs) | **get** /projects/{id}/configs | List Configs
[**list_deployed_configs**](#list_deployed_configs) | **get** /projects/{id}/deployed-configs | List Deployed Configs
[**update**](#update) | **patch** /projects/{id} | Update
[**update_feedback_types**](#update_feedback_types) | **patch** /projects/{id}/feedback-types | Update Feedback Types

# **create**

Create a new project.

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
    create_response = humanloop.projects.create(
        name="string_example",  # required
        feedback_types=[
            {
                "type": "type_example",
            }
        ],  # optional
    )
    pprint(create_response.body)
    pprint(create_response.body["id"])
    pprint(create_response.body["name"])
    pprint(create_response.body["users"])
    pprint(create_response.body["data_count"])
    pprint(create_response.body["feedback_types"])
    pprint(create_response.body["team_id"])
    pprint(create_response.body["created_at"])
    pprint(create_response.body["updated_at"])
    pprint(create_response.body["active_experiment"])
    pprint(create_response.body["active_config"])
    pprint(create_response.body["config_type"])
    pprint(create_response.body["active_evaluators"])
    pprint(create_response.headers)
    pprint(create_response.status)
    pprint(create_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ProjectsApi.create: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_feedback_type**

Create Feedback Type

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
    # Create Feedback Type
    create_feedback_type_response = humanloop.projects.create_feedback_type(
        type="string_example",  # required
        id="id_example",  # required
        values=[
            {
                "value": "value_example",
                "sentiment": "positive",
            }
        ],  # optional
        _class="string_example",  # optional
    )
    pprint(create_feedback_type_response.body)
    pprint(create_feedback_type_response.body["type"])
    pprint(create_feedback_type_response.body["values"])
    pprint(create_feedback_type_response.headers)
    pprint(create_feedback_type_response.status)
    pprint(create_feedback_type_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ProjectsApi.create_feedback_type: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **deactivate_config**

Remove the project's active config, if set.  This has no effect if the project does not have an active model config set.

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
    # Deactivate Config
    deactivate_config_response = humanloop.projects.deactivate_config(
        id="id_example",  # required
        environment="string_example",  # optional
    )
    pprint(deactivate_config_response.body)
    pprint(deactivate_config_response.body["id"])
    pprint(deactivate_config_response.body["name"])
    pprint(deactivate_config_response.body["users"])
    pprint(deactivate_config_response.body["data_count"])
    pprint(deactivate_config_response.body["feedback_types"])
    pprint(deactivate_config_response.body["team_id"])
    pprint(deactivate_config_response.body["created_at"])
    pprint(deactivate_config_response.body["updated_at"])
    pprint(deactivate_config_response.body["active_experiment"])
    pprint(deactivate_config_response.body["active_config"])
    pprint(deactivate_config_response.body["config_type"])
    pprint(deactivate_config_response.body["active_evaluators"])
    pprint(deactivate_config_response.headers)
    pprint(deactivate_config_response.status)
    pprint(deactivate_config_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ProjectsApi.deactivate_config: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **deactivate_experiment**

Remove the project's active experiment, if set.  This has no effect if the project does not have an active experiment set.

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
    # Deactivate Experiment
    deactivate_experiment_response = humanloop.projects.deactivate_experiment(
        id="id_example",  # required
        environment="string_example",  # optional
    )
    pprint(deactivate_experiment_response.body)
    pprint(deactivate_experiment_response.body["id"])
    pprint(deactivate_experiment_response.body["name"])
    pprint(deactivate_experiment_response.body["users"])
    pprint(deactivate_experiment_response.body["data_count"])
    pprint(deactivate_experiment_response.body["feedback_types"])
    pprint(deactivate_experiment_response.body["team_id"])
    pprint(deactivate_experiment_response.body["created_at"])
    pprint(deactivate_experiment_response.body["updated_at"])
    pprint(deactivate_experiment_response.body["active_experiment"])
    pprint(deactivate_experiment_response.body["active_config"])
    pprint(deactivate_experiment_response.body["config_type"])
    pprint(deactivate_experiment_response.body["active_evaluators"])
    pprint(deactivate_experiment_response.headers)
    pprint(deactivate_experiment_response.status)
    pprint(deactivate_experiment_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ProjectsApi.deactivate_experiment: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **delete_deployed_config**

Remove the model config deployed to environment.  This has no effect if the project does not have an active model config set.

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
    # Delete Deployed Config
    delete_deployed_config_response = humanloop.projects.delete_deployed_config(
        project_id="project_id_example",  # required
        environment_id="environment_id_example",  # required
    )
    pprint(delete_deployed_config_response.headers)
    pprint(delete_deployed_config_response.status)
    pprint(delete_deployed_config_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ProjectsApi.delete_deployed_config: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **deploy_config**

Deploy a model config to an environment.  If the environment already has a model config deployed, it will be replaced.

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
    # Deploy Config
    deploy_config_response = humanloop.projects.deploy_config(
        project_id="project_id_example",  # required
        config_id="string_example",  # optional
        experiment_id="string_example",  # optional
        environments=[
            {
                "id": "id_example",
            }
        ],  # optional
    )
    pprint(deploy_config_response.body)
    pprint(deploy_config_response.headers)
    pprint(deploy_config_response.status)
    pprint(deploy_config_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ProjectsApi.deploy_config: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **export**

Export all logged datapoints associated to your project.  Results are paginated and sorts the datapoints based on `created_at` in descending order.

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
    # Export
    export_response = humanloop.projects.export(
        id="id_example",  # required
        page=0,  # optional
        size=10,  # optional
    )
    pprint(export_response.headers)
    pprint(export_response.status)
    pprint(export_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ProjectsApi.export: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get**

Get a specific project.

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
    get_response = humanloop.projects.get(
        id="id_example",  # required
    )
    pprint(get_response.body)
    pprint(get_response.body["id"])
    pprint(get_response.body["name"])
    pprint(get_response.body["users"])
    pprint(get_response.body["data_count"])
    pprint(get_response.body["feedback_types"])
    pprint(get_response.body["team_id"])
    pprint(get_response.body["created_at"])
    pprint(get_response.body["updated_at"])
    pprint(get_response.body["active_experiment"])
    pprint(get_response.body["active_config"])
    pprint(get_response.body["config_type"])
    pprint(get_response.body["active_evaluators"])
    pprint(get_response.headers)
    pprint(get_response.status)
    pprint(get_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ProjectsApi.get: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_active_config**

Retrieves a config to use to execute your model.  A config will be selected based on the project's active config/experiment settings.

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
    # Get Active Config
    get_active_config_response = humanloop.projects.get_active_config(
        id="id_example",  # required
        environment="string_example",  # optional
    )
    pprint(get_active_config_response.body)
    pprint(get_active_config_response.body["project_id"])
    pprint(get_active_config_response.body["project_name"])
    pprint(get_active_config_response.body["created_at"])
    pprint(get_active_config_response.body["updated_at"])
    pprint(get_active_config_response.body["last_used"])
    pprint(get_active_config_response.body["config"])
    pprint(get_active_config_response.body["feedback_stats"])
    pprint(get_active_config_response.body["num_datapoints"])
    pprint(get_active_config_response.body["experiment_id"])
    pprint(get_active_config_response.body["evaluation_aggregates"])
    pprint(get_active_config_response.body["trial_id"])
    pprint(get_active_config_response.headers)
    pprint(get_active_config_response.status)
    pprint(get_active_config_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ProjectsApi.get_active_config: %s\n" % e)
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

Get a paginated list of projects.

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
    list_response = humanloop.projects.list(
        page=0,  # optional
        size=10,  # optional
        organization_id="string_example",  # optional
        filter="string_example",  # optional
        user_filter="string_example",  # optional
        sort_by="string_example",  # optional
        order="string_example",  # optional
    )
    pprint(list_response.headers)
    pprint(list_response.status)
    pprint(list_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ProjectsApi.list: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **list_configs**

Get an array of configs associated to your project.

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
    # List Configs
    list_configs_response = humanloop.projects.list_configs(
        id="id_example",  # required
        evaluation_aggregates=True,  # optional
    )
    pprint(list_configs_response.body)
    pprint(list_configs_response.headers)
    pprint(list_configs_response.status)
    pprint(list_configs_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ProjectsApi.list_configs: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **list_deployed_configs**

Get an array of environments with the deployed configs associated to your project.

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
    # List Deployed Configs
    list_deployed_configs_response = humanloop.projects.list_deployed_configs(
        id="id_example",  # required
    )
    pprint(list_deployed_configs_response.body)
    pprint(list_deployed_configs_response.headers)
    pprint(list_deployed_configs_response.status)
    pprint(list_deployed_configs_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ProjectsApi.list_deployed_configs: %s\n" % e)
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

Update a specific project.  Set the project's active model config/experiment by passing either `active_experiment_id` or `active_model_config_id`. These will be set to the Default environment unless a list of environments are also passed in specifically detailing which environments to assign the active config or experiment.  Set the feedback labels to be treated as positive user feedback used in calculating top-level project metrics by passing a list of labels in `positive_labels`.

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
    update_response = humanloop.projects.update(
        id="id_example",  # required
        name="string_example",  # optional
        active_experiment_id="string_example",  # optional
        active_config_id="string_example",  # optional
        positive_labels=[
            {
                "type": "type_example",
                "value": "value_example",
            }
        ],  # optional
    )
    pprint(update_response.body)
    pprint(update_response.body["id"])
    pprint(update_response.body["name"])
    pprint(update_response.body["users"])
    pprint(update_response.body["data_count"])
    pprint(update_response.body["feedback_types"])
    pprint(update_response.body["team_id"])
    pprint(update_response.body["created_at"])
    pprint(update_response.body["updated_at"])
    pprint(update_response.body["active_experiment"])
    pprint(update_response.body["active_config"])
    pprint(update_response.body["config_type"])
    pprint(update_response.body["active_evaluators"])
    pprint(update_response.headers)
    pprint(update_response.status)
    pprint(update_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ProjectsApi.update: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **update_feedback_types**

Update feedback types.  Allows enabling the available feedback types and setting status of feedback types/categorical values.  This behaves like an upsert; any feedback categorical values that do not already exist in the project will be created.

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
    # Update Feedback Types
    update_feedback_types_response = humanloop.projects.update_feedback_types(
        body=[
            {
                "type": "type_example",
            }
        ],  # required
        id="id_example",  # required
    )
    pprint(update_feedback_types_response.body)
    pprint(update_feedback_types_response.headers)
    pprint(update_feedback_types_response.status)
    pprint(update_feedback_types_response.round_trip_time)
except ApiException as e:
    print("Exception when calling ProjectsApi.update_feedback_types: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

