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


class DashboardConfiguration(
    schemas.DictSchema
):
    """
    This class is auto generated by Konfig (https://konfigthis.com)
    """


    class MetaOapg:
        required = {
            "time_range_days",
            "model_config_ids",
            "time_unit",
        }
        
        class properties:
        
            @staticmethod
            def time_unit() -> typing.Type['TimeUnit']:
                return TimeUnit
            time_range_days = schemas.IntSchema
        
            @staticmethod
            def model_config_ids() -> typing.Type['DashboardConfigurationModelConfigIds']:
                return DashboardConfigurationModelConfigIds
            __annotations__ = {
                "time_unit": time_unit,
                "time_range_days": time_range_days,
                "model_config_ids": model_config_ids,
            }
    
    time_range_days: MetaOapg.properties.time_range_days
    model_config_ids: 'DashboardConfigurationModelConfigIds'
    time_unit: 'TimeUnit'
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["time_unit"]) -> 'TimeUnit': ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["time_range_days"]) -> MetaOapg.properties.time_range_days: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["model_config_ids"]) -> 'DashboardConfigurationModelConfigIds': ...
    
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    
    def __getitem__(self, name: typing.Union[typing_extensions.Literal["time_unit", "time_range_days", "model_config_ids", ], str]):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["time_unit"]) -> 'TimeUnit': ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["time_range_days"]) -> MetaOapg.properties.time_range_days: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["model_config_ids"]) -> 'DashboardConfigurationModelConfigIds': ...
    
    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    
    def get_item_oapg(self, name: typing.Union[typing_extensions.Literal["time_unit", "time_range_days", "model_config_ids", ], str]):
        return super().get_item_oapg(name)
    

    def __new__(
        cls,
        *args: typing.Union[dict, frozendict.frozendict, ],
        time_range_days: typing.Union[MetaOapg.properties.time_range_days, decimal.Decimal, int, ],
        model_config_ids: 'DashboardConfigurationModelConfigIds',
        time_unit: 'TimeUnit',
        _configuration: typing.Optional[schemas.Configuration] = None,
        **kwargs: typing.Union[schemas.AnyTypeSchema, dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, None, list, tuple, bytes],
    ) -> 'DashboardConfiguration':
        return super().__new__(
            cls,
            *args,
            time_range_days=time_range_days,
            model_config_ids=model_config_ids,
            time_unit=time_unit,
            _configuration=_configuration,
            **kwargs,
        )

from humanloop.model.dashboard_configuration_model_config_ids import DashboardConfigurationModelConfigIds
from humanloop.model.time_unit import TimeUnit
