# humanloop.logs

All URIs are relative to *https://api.humanloop.com/v4*

Method | HTTP request | Description
------------- | ------------- | -------------
[**log**](#log) | **post** /logs | Log
[**update**](#update) | **patch** /logs/{id} | Update
[**update_by_ref**](#update_by_ref) | **patch** /logs | Update By Reference Id

# **log**

Log a datapoint or array of datapoints to your Humanloop project.

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
    # Log
    log_response = humanloop.log(
        body=[{}],  # required
        project="string_example",  # optional
        project_id="string_example",  # optional
        session_id="string_example",  # optional
        session_reference_id="string_example",  # optional
        parent_id="string_example",  # optional
        parent_reference_id="string_example",  # optional
        inputs={},  # optional
        source="string_example",  # optional
        metadata={},  # optional
        reference_id="string_example",  # optional
        trial_id="string_example",  # optional
        messages=[
            {
                "role": "user",
            }
        ],  # optional
        output="string_example",  # optional
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
        },  # optional
        feedback={
            "type": "string_example",
            "value": 3.14,
        },  # optional
        created_at="1970-01-01T00:00:00.00Z",  # optional
        error="string_example",  # optional
        duration=3.14,  # optional
    )
    pprint(log_response.body)
    pprint(log_response.headers)
    pprint(log_response.status)
    pprint(log_response.round_trip_time)
except ApiException as e:
    print("Exception when calling LogsApi.log: %s\n" % e)
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

Update a logged datapoint in your Humanloop project.

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
    update_response = humanloop.logs.update(
        id="id_example",  # required
        output="string_example",  # optional
        error="string_example",  # optional
        duration=3.14,  # optional
    )
    pprint(update_response.body)
    pprint(update_response.body["config"])
    pprint(update_response.body["id"])
    pprint(update_response.body["project"])
    pprint(update_response.body["project_id"])
    pprint(update_response.body["session_id"])
    pprint(update_response.body["session_reference_id"])
    pprint(update_response.body["parent_id"])
    pprint(update_response.body["parent_reference_id"])
    pprint(update_response.body["inputs"])
    pprint(update_response.body["source"])
    pprint(update_response.body["metadata"])
    pprint(update_response.body["reference_id"])
    pprint(update_response.body["trial_id"])
    pprint(update_response.body["messages"])
    pprint(update_response.body["output"])
    pprint(update_response.body["feedback"])
    pprint(update_response.body["created_at"])
    pprint(update_response.body["error"])
    pprint(update_response.body["duration"])
    pprint(update_response.body["model_config"])
    pprint(update_response.body["user"])
    pprint(update_response.body["provider_response"])
    pprint(update_response.body["provider_latency"])
    pprint(update_response.body["raw_output"])
    pprint(update_response.body["finish_reason"])
    pprint(update_response.headers)
    pprint(update_response.status)
    pprint(update_response.round_trip_time)
except ApiException as e:
    print("Exception when calling LogsApi.update: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **update_by_ref**

Update a logged datapoint by its reference ID.  The `reference_id` query parameter must be provided, and refers to the `reference_id` of a previously-logged datapoint.

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
    # Update By Reference Id
    update_by_ref_response = humanloop.logs.update_by_ref(
        reference_id="reference_id_example",  # required
        output="string_example",  # optional
        error="string_example",  # optional
        duration=3.14,  # optional
    )
    pprint(update_by_ref_response.body)
    pprint(update_by_ref_response.body["config"])
    pprint(update_by_ref_response.body["id"])
    pprint(update_by_ref_response.body["project"])
    pprint(update_by_ref_response.body["project_id"])
    pprint(update_by_ref_response.body["session_id"])
    pprint(update_by_ref_response.body["session_reference_id"])
    pprint(update_by_ref_response.body["parent_id"])
    pprint(update_by_ref_response.body["parent_reference_id"])
    pprint(update_by_ref_response.body["inputs"])
    pprint(update_by_ref_response.body["source"])
    pprint(update_by_ref_response.body["metadata"])
    pprint(update_by_ref_response.body["reference_id"])
    pprint(update_by_ref_response.body["trial_id"])
    pprint(update_by_ref_response.body["messages"])
    pprint(update_by_ref_response.body["output"])
    pprint(update_by_ref_response.body["feedback"])
    pprint(update_by_ref_response.body["created_at"])
    pprint(update_by_ref_response.body["error"])
    pprint(update_by_ref_response.body["duration"])
    pprint(update_by_ref_response.body["model_config"])
    pprint(update_by_ref_response.body["user"])
    pprint(update_by_ref_response.body["provider_response"])
    pprint(update_by_ref_response.body["provider_latency"])
    pprint(update_by_ref_response.body["raw_output"])
    pprint(update_by_ref_response.body["finish_reason"])
    pprint(update_by_ref_response.headers)
    pprint(update_by_ref_response.status)
    pprint(update_by_ref_response.round_trip_time)
except ApiException as e:
    print("Exception when calling LogsApi.update_by_ref: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

