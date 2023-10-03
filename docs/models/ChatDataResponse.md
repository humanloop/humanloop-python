# humanloop.model.chat_data_response.ChatDataResponse

Overwrite DataResponse for chat.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Overwrite DataResponse for chat. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**output** | str,  | str,  | Output text returned from the provider model with leading and trailing whitespaces stripped. | 
**model_config_id** | str,  | str,  | The model configuration used to create the generation. | 
**raw_output** | str,  | str,  | Raw output text returned from the provider model. | 
**index** | decimal.Decimal, int,  | decimal.Decimal,  | The index for the sampled generation for a given input. The num_samples request parameter controls how many samples are generated. | 
**id** | str,  | str,  | Unique ID for the model inputs and output logged to Humanloop. Use this when recording feedback later. | 
**[inputs](#inputs)** | dict, frozendict.frozendict,  | frozendict.frozendict,  | The inputs passed to the chat template. | [optional] 
**finish_reason** | str,  | str,  | Why the generation ended. One of &#x27;stop&#x27; (indicating a stop token was encountered), or &#x27;length&#x27; (indicating the max tokens limit has been reached), or &#x27;tool_call&#x27; (indicating that the model has chosen to call a tool - in which case the tool_call parameter of the response will be populated). It will be set as null for the intermediary responses during a stream, and will only be set as non-null for the final streamed token. | [optional] 
**[tool_results](#tool_results)** | list, tuple,  | tuple,  | Results of any tools run during the generation. | [optional] 
**[messages](#messages)** | list, tuple,  | tuple,  | The messages passed to the to provider chat endpoint. | [optional] 
**[tool_call](#tool_call)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | JSON definition of the tool to call and the corresponding argument values. Will be populated when finish_reason&#x3D;&#x27;tool_call&#x27;. | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# inputs

The inputs passed to the chat template.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | The inputs passed to the chat template. | 

# tool_results

Results of any tools run during the generation.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  | Results of any tools run during the generation. | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**ToolResultResponse**](ToolResultResponse.md) | [**ToolResultResponse**](ToolResultResponse.md) | [**ToolResultResponse**](ToolResultResponse.md) |  | 

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

# tool_call

JSON definition of the tool to call and the corresponding argument values. Will be populated when finish_reason='tool_call'.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | JSON definition of the tool to call and the corresponding argument values. Will be populated when finish_reason&#x3D;&#x27;tool_call&#x27;. | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### allOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[ToolCall](ToolCall.md) | [**ToolCall**](ToolCall.md) | [**ToolCall**](ToolCall.md) |  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

