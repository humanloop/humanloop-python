# coding: utf-8

"""
    Humanloop API

    The Humanloop API allows you to interact with Humanloop from your product or service.  You can do this through HTTP requests from any language or via our official Python or TypeScript SDK.  To install the official [Python SDK](https://pypi.org/project/humanloop/), run the following command:  ```bash pip install humanloop ```  To install the official [TypeScript SDK](https://www.npmjs.com/package/humanloop), run the following command:  ```bash npm i humanloop ```  ---  Guides and further details about key concepts can be found in [our docs](https://docs.humanloop.com/).

    The version of the OpenAPI document: 4.0.1
    Generated by: https://konfigthis.com
"""

from datetime import date, datetime  # noqa: F401
import decimal  # noqa: F401
import functools  # noqa: F401
import io  # noqa: F401
import re  # noqa: F401
import typing  # noqa: F401
import typing_extensions  # noqa: F401
import uuid  # noqa: F401

import frozendict  # noqa: F401

from humanloop import schemas  # noqa: F401


class DatapointResponse(
    schemas.DictSchema
):
    """
    This class is auto generated by Konfig (https://konfigthis.com)
    """


    class MetaOapg:
        required = {
            "updated_at",
            "dataset_id",
            "created_at",
            "id",
        }
        
        class properties:
            id = schemas.StrSchema
            dataset_id = schemas.StrSchema
            created_at = schemas.DateTimeSchema
            updated_at = schemas.DateTimeSchema
        
            @staticmethod
            def inputs() -> typing.Type['DatapointResponseInputs']:
                return DatapointResponseInputs
            
            
            class messages(
                schemas.ListSchema
            ):
            
            
                class MetaOapg:
                    
                    @staticmethod
                    def items() -> typing.Type['ChatMessageWithToolCall']:
                        return ChatMessageWithToolCall
            
                def __new__(
                    cls,
                    arg: typing.Union[typing.Tuple['ChatMessageWithToolCall'], typing.List['ChatMessageWithToolCall']],
                    _configuration: typing.Optional[schemas.Configuration] = None,
                ) -> 'messages':
                    return super().__new__(
                        cls,
                        arg,
                        _configuration=_configuration,
                    )
            
                def __getitem__(self, i: int) -> 'ChatMessageWithToolCall':
                    return super().__getitem__(i)
        
            @staticmethod
            def target() -> typing.Type['DatapointResponseTarget']:
                return DatapointResponseTarget
            source_project_data_id = schemas.StrSchema
            __annotations__ = {
                "id": id,
                "dataset_id": dataset_id,
                "created_at": created_at,
                "updated_at": updated_at,
                "inputs": inputs,
                "messages": messages,
                "target": target,
                "source_project_data_id": source_project_data_id,
            }
    
    updated_at: MetaOapg.properties.updated_at
    dataset_id: MetaOapg.properties.dataset_id
    created_at: MetaOapg.properties.created_at
    id: MetaOapg.properties.id
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["id"]) -> MetaOapg.properties.id: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["dataset_id"]) -> MetaOapg.properties.dataset_id: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["created_at"]) -> MetaOapg.properties.created_at: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["updated_at"]) -> MetaOapg.properties.updated_at: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["inputs"]) -> 'DatapointResponseInputs': ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["messages"]) -> MetaOapg.properties.messages: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["target"]) -> 'DatapointResponseTarget': ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["source_project_data_id"]) -> MetaOapg.properties.source_project_data_id: ...
    
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    
    def __getitem__(self, name: typing.Union[typing_extensions.Literal["id", "dataset_id", "created_at", "updated_at", "inputs", "messages", "target", "source_project_data_id", ], str]):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["id"]) -> MetaOapg.properties.id: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["dataset_id"]) -> MetaOapg.properties.dataset_id: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["created_at"]) -> MetaOapg.properties.created_at: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["updated_at"]) -> MetaOapg.properties.updated_at: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["inputs"]) -> typing.Union['DatapointResponseInputs', schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["messages"]) -> typing.Union[MetaOapg.properties.messages, schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["target"]) -> typing.Union['DatapointResponseTarget', schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["source_project_data_id"]) -> typing.Union[MetaOapg.properties.source_project_data_id, schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    
    def get_item_oapg(self, name: typing.Union[typing_extensions.Literal["id", "dataset_id", "created_at", "updated_at", "inputs", "messages", "target", "source_project_data_id", ], str]):
        return super().get_item_oapg(name)
    

    def __new__(
        cls,
        *args: typing.Union[dict, frozendict.frozendict, ],
        updated_at: typing.Union[MetaOapg.properties.updated_at, str, datetime, ],
        dataset_id: typing.Union[MetaOapg.properties.dataset_id, str, ],
        created_at: typing.Union[MetaOapg.properties.created_at, str, datetime, ],
        id: typing.Union[MetaOapg.properties.id, str, ],
        inputs: typing.Union['DatapointResponseInputs', schemas.Unset] = schemas.unset,
        messages: typing.Union[MetaOapg.properties.messages, list, tuple, schemas.Unset] = schemas.unset,
        target: typing.Union['DatapointResponseTarget', schemas.Unset] = schemas.unset,
        source_project_data_id: typing.Union[MetaOapg.properties.source_project_data_id, str, schemas.Unset] = schemas.unset,
        _configuration: typing.Optional[schemas.Configuration] = None,
        **kwargs: typing.Union[schemas.AnyTypeSchema, dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, None, list, tuple, bytes],
    ) -> 'DatapointResponse':
        return super().__new__(
            cls,
            *args,
            updated_at=updated_at,
            dataset_id=dataset_id,
            created_at=created_at,
            id=id,
            inputs=inputs,
            messages=messages,
            target=target,
            source_project_data_id=source_project_data_id,
            _configuration=_configuration,
            **kwargs,
        )

from humanloop.model.chat_message_with_tool_call import ChatMessageWithToolCall
from humanloop.model.datapoint_response_inputs import DatapointResponseInputs
from humanloop.model.datapoint_response_target import DatapointResponseTarget
