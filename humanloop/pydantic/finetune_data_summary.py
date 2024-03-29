# coding: utf-8

"""
    Humanloop API

    The Humanloop API allows you to interact with Humanloop from your product or service.  You can do this through HTTP requests from any language or via our official Python or TypeScript SDK.  To install the official [Python SDK](https://pypi.org/project/humanloop/), run the following command:  ```bash pip install humanloop ```  To install the official [TypeScript SDK](https://www.npmjs.com/package/humanloop), run the following command:  ```bash npm i humanloop ```  ---  Guides and further details about key concepts can be found in [our docs](https://docs.humanloop.com/).

    The version of the OpenAPI document: 4.0.1
    Generated by: https://konfigthis.com
"""

from datetime import datetime, date
import typing
from enum import Enum
from typing_extensions import TypedDict, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field, RootModel, ConfigDict

from humanloop.pydantic.finetune_data_summary_errors import FinetuneDataSummaryErrors

class FinetuneDataSummary(BaseModel):
    data_count: int = Field(alias='data_count')

    error_count: int = Field(alias='error_count')

    truncated_data_count: int = Field(alias='truncated_data_count')

    token_count: int = Field(alias='token_count')

    truncated_token_count: int = Field(alias='truncated_token_count')

    cost: typing.Union[int, float] = Field(alias='cost')

    errors: FinetuneDataSummaryErrors = Field(alias='errors')

    dataset_name: str = Field(alias='dataset_name')

    dataset_id: str = Field(alias='dataset_id')

    model_config = ConfigDict(
        protected_namespaces=(),
        arbitrary_types_allowed=True
    )
