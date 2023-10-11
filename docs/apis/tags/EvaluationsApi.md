# humanloop.evaluations

All URIs are relative to *https://api.humanloop.com/v4*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create**](#create) | **post** /projects/{project_id}/evaluations | Create
[**get**](#get) | **get** /evaluations/{id} | Get
[**list_all_for_project**](#list_all_for_project) | **get** /projects/{project_id}/evaluations | List For Project
[**list_testcases**](#list_testcases) | **get** /evaluations/{id}/testcases | List Testcases

# **create**

Create an evaluation.

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
    create_response = humanloop.evaluations.create(
        config_id="string_example",  # required
        evaluator_ids=["string_example"],  # required
        testset_id="string_example",  # required
        project_id="project_id_example",  # required
        provider_api_keys={},  # optional
    )
    pprint(create_response.body)
    pprint(create_response.body["id"])
    pprint(create_response.body["status"])
    pprint(create_response.body["config"])
    pprint(create_response.body["created_at"])
    pprint(create_response.body["updated_at"])
    pprint(create_response.body["evaluators"])
    pprint(create_response.body["testset"])
    pprint(create_response.body["testset_snapshot"])
    pprint(create_response.body["evaluator_aggregates"])
    pprint(create_response.headers)
    pprint(create_response.status)
    pprint(create_response.round_trip_time)
except ApiException as e:
    print("Exception when calling EvaluationsApi.create: %s\n" % e)
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

Get evaluation by ID.

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
    get_response = humanloop.evaluations.get(
        id="id_example",  # required
        evaluator_aggregates=True,  # optional
    )
    pprint(get_response.body)
    pprint(get_response.body["id"])
    pprint(get_response.body["status"])
    pprint(get_response.body["config"])
    pprint(get_response.body["created_at"])
    pprint(get_response.body["updated_at"])
    pprint(get_response.body["evaluators"])
    pprint(get_response.body["testset"])
    pprint(get_response.body["testset_snapshot"])
    pprint(get_response.body["evaluator_aggregates"])
    pprint(get_response.headers)
    pprint(get_response.status)
    pprint(get_response.round_trip_time)
except ApiException as e:
    print("Exception when calling EvaluationsApi.get: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **list_all_for_project**

Get all the evaluations associated with your project.

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
    # List For Project
    list_all_for_project_response = humanloop.evaluations.list_all_for_project(
        project_id="project_id_example",  # required
        evaluator_aggregates=True,  # optional
    )
    pprint(list_all_for_project_response.body)
    pprint(list_all_for_project_response.headers)
    pprint(list_all_for_project_response.status)
    pprint(list_all_for_project_response.round_trip_time)
except ApiException as e:
    print("Exception when calling EvaluationsApi.list_all_for_project: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **list_testcases**

Get testcases by evaluation ID.

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
    # List Testcases
    list_testcases_response = humanloop.evaluations.list_testcases(
        id="id_example",  # required
        page=1,  # optional
        size=10,  # optional
    )
    pprint(list_testcases_response.headers)
    pprint(list_testcases_response.status)
    pprint(list_testcases_response.round_trip_time)
except ApiException as e:
    print("Exception when calling EvaluationsApi.list_testcases: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

