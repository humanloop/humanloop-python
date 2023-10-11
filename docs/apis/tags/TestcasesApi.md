# humanloop.testcases

All URIs are relative to *https://api.humanloop.com/v4*

Method | HTTP request | Description
------------- | ------------- | -------------
[**delete**](#delete) | **delete** /testcases | Delete
[**get**](#get) | **get** /testcases/{id} | Get
[**update**](#update) | **patch** /testcases/{id} | Update

# **delete**

Delete a list of testsets by their IDs.

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
    humanloop.testcases.delete(
        body=["testsets_delete_testcase_request_example"],  # required
    )
except ApiException as e:
    print("Exception when calling TestcasesApi.delete: %s\n" % e)
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

Get a testcase by ID.

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
    get_response = humanloop.testcases.get(
        id="id_example",  # required
    )
    pprint(get_response.body)
    pprint(get_response.body["id"])
    pprint(get_response.body["testset_id"])
    pprint(get_response.body["created_at"])
    pprint(get_response.body["updated_at"])
    pprint(get_response.body["inputs"])
    pprint(get_response.body["messages"])
    pprint(get_response.body["target"])
    pprint(get_response.body["source_project_data_id"])
    pprint(get_response.headers)
    pprint(get_response.status)
    pprint(get_response.round_trip_time)
except ApiException as e:
    print("Exception when calling TestcasesApi.get: %s\n" % e)
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

Edit the input, messages and criteria fields of a testcase. The fields passed in the request are the ones edited.  Passing `null` as a value for a field will delete that field. In order to signify not changing a field, it should be omitted from the request body.

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
    update_response = humanloop.testcases.update(
        id="id_example",  # required
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
    pprint(update_response.body)
    pprint(update_response.body["id"])
    pprint(update_response.body["testset_id"])
    pprint(update_response.body["created_at"])
    pprint(update_response.body["updated_at"])
    pprint(update_response.body["inputs"])
    pprint(update_response.body["messages"])
    pprint(update_response.body["target"])
    pprint(update_response.body["source_project_data_id"])
    pprint(update_response.headers)
    pprint(update_response.status)
    pprint(update_response.round_trip_time)
except ApiException as e:
    print("Exception when calling TestcasesApi.update: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

