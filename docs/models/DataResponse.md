# humanloop.model.data_response.DataResponse

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**output** | str,  | str,  | Output text returned from the provider model with leading and trailing whitespaces stripped. | 
**[inputs](#inputs)** | dict, frozendict.frozendict,  | frozendict.frozendict,  | The inputs passed to the prompt template. | 
**model_config_id** | str,  | str,  | The model configuration used to create the generation. | 
**raw_output** | str,  | str,  | Raw output text returned from the provider model. | 
**index** | decimal.Decimal, int,  | decimal.Decimal,  | The index for the sampled generation for a given input. The num_samples request parameter controls how many samples are generated. | 
**id** | str,  | str,  | Unique ID for the model inputs and output logged to Humanloop. Use this when recording feedback later. | 
**finish_reason** | str,  | str,  | Why the generation ended. One of &#x27;stop&#x27; (indicating a stop token was encountered), or &#x27;length&#x27; (indicating the max tokens limit has been reached), or &#x27;tool_call&#x27; (indicating that the model has chosen to call a tool - in which case the tool_call parameter of the response will be populated). It will be set as null for the intermediary responses during a stream, and will only be set as non-null for the final streamed token. | [optional] 
**[tool_results](#tool_results)** | list, tuple,  | tuple,  | Results of any tools run during the generation. | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# inputs

The inputs passed to the prompt template.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | The inputs passed to the prompt template. | 

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

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

