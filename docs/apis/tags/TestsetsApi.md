# humanloop.testsets

All URIs are relative to *https://api.humanloop.com/v4*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create**](#create) | **post** /projects/{project_id}/testsets | Create
[**create_testcase**](#create_testcase) | **post** /testsets/{testset_id}/testcases | Create Testcase
[**delete**](#delete) | **delete** /testsets/{id} | Delete
[**get**](#get) | **get** /testsets/{id} | Get
[**list_all_for_project**](#list_all_for_project) | **get** /projects/{project_id}/testsets | List For Project
[**list_testcases**](#list_testcases) | **get** /testsets/{testset_id}/testcases | List Testcases
[**update**](#update) | **patch** /testsets/{id} | Update

# **create**

Create a new testset for a project.

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
    create_response = humanloop.testsets.create(
        description="string_example",  # required
        name="string_example",  # required
        project_id="project_id_example",  # required
    )
    pprint(create_response.body)
    pprint(create_response.body["id"])
    pprint(create_response.body["project_id"])
    pprint(create_response.body["name"])
    pprint(create_response.body["testcase_count"])
    pprint(create_response.body["created_at"])
    pprint(create_response.body["updated_at"])
    pprint(create_response.body["description"])
    pprint(create_response.headers)
    pprint(create_response.status)
    pprint(create_response.round_trip_time)
except ApiException as e:
    print("Exception when calling TestsetsApi.create: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_testcase**

Create a new testcase for a testset.

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
    # Create Testcase
    create_testcase_response = humanloop.testsets.create_testcase(
        body={
            "datapoint_ids": ["datapoint_ids_example"],
        },  # required
        testset_id="testset_id_example",  # required
        datapoint_ids=["string_example"],  # optional
        inputs={
            "key": "string_example",
        },  # optional
        messages=[
            {
                "role": "user",
            }
        ],  # optional
        target={
            "key": {},
        },  # optional
    )
    pprint(create_testcase_response.body)
    pprint(create_testcase_response.headers)
    pprint(create_testcase_response.status)
    pprint(create_testcase_response.round_trip_time)
except ApiException as e:
    print("Exception when calling TestsetsApi.create_testcase: %s\n" % e)
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

Delete a testset by ID.

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
    delete_response = humanloop.testsets.delete(
        id="id_example",  # required
    )
    pprint(delete_response.body)
    pprint(delete_response.body["id"])
    pprint(delete_response.body["project_id"])
    pprint(delete_response.body["name"])
    pprint(delete_response.body["testcase_count"])
    pprint(delete_response.body["created_at"])
    pprint(delete_response.body["updated_at"])
    pprint(delete_response.body["description"])
    pprint(delete_response.headers)
    pprint(delete_response.status)
    pprint(delete_response.round_trip_time)
except ApiException as e:
    print("Exception when calling TestsetsApi.delete: %s\n" % e)
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

Get a single testset by ID.

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
    get_response = humanloop.testsets.get(
        id="id_example",  # required
    )
    pprint(get_response.body)
    pprint(get_response.body["id"])
    pprint(get_response.body["project_id"])
    pprint(get_response.body["name"])
    pprint(get_response.body["testcase_count"])
    pprint(get_response.body["created_at"])
    pprint(get_response.body["updated_at"])
    pprint(get_response.body["description"])
    pprint(get_response.headers)
    pprint(get_response.status)
    pprint(get_response.round_trip_time)
except ApiException as e:
    print("Exception when calling TestsetsApi.get: %s\n" % e)
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

Get all testsets for a project.

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
    list_all_for_project_response = humanloop.testsets.list_all_for_project(
        project_id="project_id_example",  # required
    )
    pprint(list_all_for_project_response.body)
    pprint(list_all_for_project_response.headers)
    pprint(list_all_for_project_response.status)
    pprint(list_all_for_project_response.round_trip_time)
except ApiException as e:
    print("Exception when calling TestsetsApi.list_all_for_project: %s\n" % e)
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

Get testcases for a testset.

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
    list_testcases_response = humanloop.testsets.list_testcases(
        testset_id="testset_id_example",  # required
        page=0,  # optional
        size=50,  # optional
    )
    pprint(list_testcases_response.headers)
    pprint(list_testcases_response.status)
    pprint(list_testcases_response.round_trip_time)
except ApiException as e:
    print("Exception when calling TestsetsApi.list_testcases: %s\n" % e)
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

Update a testset by ID.

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
    update_response = humanloop.testsets.update(
        id="id_example",  # required
        description="string_example",  # optional
        name="string_example",  # optional
    )
    pprint(update_response.body)
    pprint(update_response.body["id"])
    pprint(update_response.body["project_id"])
    pprint(update_response.body["name"])
    pprint(update_response.body["testcase_count"])
    pprint(update_response.body["created_at"])
    pprint(update_response.body["updated_at"])
    pprint(update_response.body["description"])
    pprint(update_response.headers)
    pprint(update_response.status)
    pprint(update_response.round_trip_time)
except ApiException as e:
    print("Exception when calling TestsetsApi.update: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

