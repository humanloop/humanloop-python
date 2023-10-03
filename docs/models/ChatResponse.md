# humanloop.model.chat_response.ChatResponse

Overwrite GenerateResponse for chat.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Overwrite GenerateResponse for chat. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**[data](#data)** | list, tuple,  | tuple,  | Array containing the chat responses. | 
**provider_responses** | [**ChatResponseProviderResponses**](ChatResponseProviderResponses.md) | [**ChatResponseProviderResponses**](ChatResponseProviderResponses.md) |  | 
**project_id** | str,  | str,  | Unique identifier of the parent project. Will not be provided if the request was made without providing a project name or id | [optional] 
**num_samples** | decimal.Decimal, int,  | decimal.Decimal,  | The number of chat responses. | [optional] if omitted the server will use the default value of 1
**logprobs** | decimal.Decimal, int,  | decimal.Decimal,  | Include the log probabilities of the top n tokens in the provider_response | [optional] 
**suffix** | str,  | str,  | The suffix that comes after a completion of inserted text. Useful for completions that act like inserts. | [optional] 
**user** | str,  | str,  | End-user ID passed through to provider call. | [optional] 
**[usage](#usage)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | Counts of the number of tokens used and related stats. | [optional] 
**[metadata](#metadata)** | dict, frozendict.frozendict,  | frozendict.frozendict,  | Any additional metadata to record. | [optional] 
**[tool_call](#tool_call)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | Controls how the model uses tools - has the same behaviour as OpenAIs function_call parameter. The following options are supported: &#x27;none&#x27; forces the model to not call a tool; the default when no tools are provided as part of the model config. &#x27;auto&#x27; the model can decide to call one of the provided tools; the default when tools are provided as part of the model config. Providing {&#x27;name&#x27;: &lt;TOOL_NAME&gt;} forces the model to use the provided tool of the same name. | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# data

Array containing the chat responses.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  | Array containing the chat responses. | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**ChatDataResponse**](ChatDataResponse.md) | [**ChatDataResponse**](ChatDataResponse.md) | [**ChatDataResponse**](ChatDataResponse.md) |  | 

# usage

Counts of the number of tokens used and related stats.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | Counts of the number of tokens used and related stats. | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### allOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[Usage](Usage.md) | [**Usage**](Usage.md) | [**Usage**](Usage.md) |  | 

# metadata

Any additional metadata to record.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Any additional metadata to record. | 

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

