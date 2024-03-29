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


class CreateEvaluationResultLogRequest(BaseModel):
    # The log that was evaluated. Must have as its `source_datapoint_id` one of the datapoints in the dataset being evaluated.
    log_id: str = Field(alias='log_id')

    # ID of the evaluator that evaluated the log. Starts with `evfn_`. Must be one of the evaluator IDs associated with the evaluation run being logged to.
    evaluator_id: str = Field(alias='evaluator_id')

    # The result value of the evaluation.
    result: typing.Optional[typing.Union[bool, int, typing.Union[int, float]]] = Field(None, alias='result')

    # An error that occurred during evaluation.
    error: typing.Optional[str] = Field(None, alias='error')

    model_config = ConfigDict(
        protected_namespaces=(),
        arbitrary_types_allowed=True
    )
