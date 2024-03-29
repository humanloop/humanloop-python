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

from humanloop.pydantic.dataset_response import DatasetResponse
from humanloop.pydantic.finetune_config import FinetuneConfig

class FinetuneResponse(BaseModel):
    # Unique identifier for fine-tuned model.
    id: str = Field(alias='id')

    # User defined friendly name for a fine-tuning run
    name: str = Field(alias='name')

    # The fine-tuning process is run async and so the resultingfine-tuned model won't be available for generations until it has completed.
    status: str = Field(alias='status')

    # Configuration details for the fine-tuned model.
    config: FinetuneConfig = Field(alias='config')

    dataset: DatasetResponse = Field(alias='dataset')

    created_at: datetime = Field(alias='created_at')

    updated_at: datetime = Field(alias='updated_at')

    # Unique reference for the fine-tuned required to make calls to the provider.
    model_name_: typing.Optional[str] = Field(None, alias='model_name')

    # Any additional metadata that you would like to log for reference.
    metadata: typing.Optional[typing.Dict[str, typing.Union[bool, date, datetime, dict, float, int, list, str, None]]] = Field(None, alias='metadata')

    # Unique ID for the fine-tuned model required to make calls to the provider's API.
    provider_id: typing.Optional[str] = Field(None, alias='provider_id')

    # Provider specific fine-tuning results.
    provider_details: typing.Optional[typing.Dict[str, typing.Union[bool, date, datetime, dict, float, int, list, str, None]]] = Field(None, alias='provider_details')

    # Summary stats about the data used for finetuning.
    data_summary: typing.Optional[typing.Dict[str, typing.Union[bool, date, datetime, dict, float, int, list, str, None]]] = Field(None, alias='data_summary')

    model_config = ConfigDict(
        protected_namespaces=(),
        arbitrary_types_allowed=True
    )
