# humanloop.feedback

All URIs are relative to *https://api.humanloop.com/v4*

Method | HTTP request | Description
------------- | ------------- | -------------
[**feedback**](#feedback) | **post** /feedback | Feedback

# **feedback**

Submit an array of feedback for existing `data_ids`

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
    # Feedback
    feedback_response = humanloop.feedback(
        body=[
            {
                "type": "string_example",
            }
        ],  # required
        type="string_example",  # optional
        value="string_example",  # optional
        data_id="string_example",  # optional
        user="string_example",  # optional
        created_at="1970-01-01T00:00:00.00Z",  # optional
        unset=True,  # optional
    )
    pprint(feedback_response.body)
    pprint(feedback_response.headers)
    pprint(feedback_response.status)
    pprint(feedback_response.round_trip_time)
except ApiException as e:
    print("Exception when calling FeedbackApi.feedback: %s\n" % e)
    pprint(e.body)
    if e.status == 422:
        pprint(e.body["detail"])
    pprint(e.headers)
    pprint(e.status)
    pprint(e.reason)
    pprint(e.round_trip_time)
```

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

