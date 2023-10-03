# humanloop.model.chat_experiment_request.ChatExperimentRequest

Base chat request.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Base chat request. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**experiment_id** | str,  | str,  | If an experiment ID is provided a model configuration will be sampled from the experiments active model configurations. | 
**[messages](#messages)** | list, tuple,  | tuple,  | The messages passed to the to provider chat endpoint. | 
**project** | str,  | str,  | Unique project name. If no project exists with this name, a new project will be created. | [optional] 
**project_id** | str,  | str,  | Unique ID of a project to associate to the log. Either this or &#x60;project&#x60; must be provided. | [optional] 
**session_id** | str,  | str,  | ID of the session to associate the datapoint. | [optional] 
**session_reference_id** | str,  | str,  | A unique string identifying the session to associate the datapoint to. Allows you to log multiple datapoints to a session (using an ID kept by your internal systems) by passing the same &#x60;session_reference_id&#x60; in subsequent log requests. Specify at most one of this or &#x60;session_id&#x60;. | [optional] 
**parent_id** | str,  | str,  | ID associated to the parent datapoint in a session. | [optional] 
**parent_reference_id** | str,  | str,  | A unique string identifying the previously-logged parent datapoint in a session. Allows you to log nested datapoints with your internal system IDs by passing the same reference ID as &#x60;parent_id&#x60; in a prior log request. Specify at most one of this or &#x60;parent_id&#x60;. Note that this cannot refer to a datapoint being logged in the same request. | [optional] 
**[inputs](#inputs)** | dict, frozendict.frozendict,  | frozendict.frozendict,  | The inputs passed to the prompt template. | [optional] if omitted the server will use the default value of {}
**source** | str,  | str,  | Identifies where the model was called from. | [optional] 
**[metadata](#metadata)** | dict, frozendict.frozendict,  | frozendict.frozendict,  | Any additional metadata to record. | [optional] 
**[provider_api_keys](#provider_api_keys)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | API keys required by each provider to make API calls. The API keys provided here are not stored by Humanloop. If not specified here, Humanloop will fall back to the key saved to your organization. | [optional] 
**num_samples** | decimal.Decimal, int,  | decimal.Decimal,  | The number of chat responses, where each chat response will use a model configuration sampled from the experiment. | [optional] if omitted the server will use the default value of 1
**stream** | bool,  | BoolClass,  | If true, tokens will be sent as data-only server-sent events. If num_samples &gt; 1, samples are streamed back independently. | [optional] if omitted the server will use the default value of False
**user** | str,  | str,  | End-user ID passed through to provider call. | [optional] 
**[tool_call](#tool_call)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | Controls how the model uses tools - has the same behaviour as OpenAIs function_call parameter. The following options are supported: &#x27;none&#x27; forces the model to not call a tool; the default when no tools are provided as part of the model config. &#x27;auto&#x27; the model can decide to call one of the provided tools; the default when tools are provided as part of the model config. Providing {&#x27;name&#x27;: &lt;TOOL_NAME&gt;} forces the model to use the provided tool of the same name. | [optional] 

# messages

The messages passed to the to provider chat endpoint.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  | The messages passed to the to provider chat endpoint. | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**ChatMessage**](ChatMessage.md) | [**ChatMessage**](ChatMessage.md) | [**ChatMessage**](ChatMessage.md) |  | 

# inputs

The inputs passed to the prompt template.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | The inputs passed to the prompt template. | if omitted the server will use the default value of {}

# metadata

Any additional metadata to record.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Any additional metadata to record. | 

# provider_api_keys

API keys required by each provider to make API calls. The API keys provided here are not stored by Humanloop. If not specified here, Humanloop will fall back to the key saved to your organization.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | API keys required by each provider to make API calls. The API keys provided here are not stored by Humanloop. If not specified here, Humanloop will fall back to the key saved to your organization. | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### allOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[ProviderApiKeys](ProviderApiKeys.md) | [**ProviderApiKeys**](ProviderApiKeys.md) | [**ProviderApiKeys**](ProviderApiKeys.md) |  | 

# tool_call

Controls how the model uses tools - has the same behaviour as OpenAIs function_call parameter. The following options are supported: 'none' forces the model to not call a tool; the default when no tools are provided as part of the model config. 'auto' the model can decide to call one of the provided tools; the default when tools are provided as part of the model config. Providing {'name': <TOOL_NAME>} forces the model to use the provided tool of the same name.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | Controls how the model uses tools - has the same behaviour as OpenAIs function_call parameter. The following options are supported: &#x27;none&#x27; forces the model to not call a tool; the default when no tools are provided as part of the model config. &#x27;auto&#x27; the model can decide to call one of the provided tools; the default when tools are provided as part of the model config. Providing {&#x27;name&#x27;: &lt;TOOL_NAME&gt;} forces the model to use the provided tool of the same name. | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### anyOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[any_of_0](#any_of_0) | str,  | str,  |  | 
[any_of_1](#any_of_1) | dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

# any_of_0

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

# any_of_1

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

