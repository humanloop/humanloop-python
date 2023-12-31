# coding: utf-8

"""
    Humanloop API

    The Humanloop API allows you to interact with Humanloop from your product or service.  You can do this through HTTP requests from any language or via our official Python or TypeScript SDK.  To install the official [Python SDK](https://pypi.org/project/humanloop/), run the following command:  ```bash pip install humanloop ```  To install the official [TypeScript SDK](https://www.npmjs.com/package/humanloop), run the following command:  ```bash npm i humanloop ```  ---  Guides and further details about key concepts can be found in [our docs](https://docs.humanloop.com/).

    The version of the OpenAPI document: 4.0.1
    Generated by: https://konfigthis.com
"""

from humanloop.paths.model_configs_deserialize.post import DeserializeRaw
from humanloop.paths.model_configs_id_export.post import ExportRaw
from humanloop.paths.model_configs_id.get import GetRaw
from humanloop.paths.model_configs.post import RegisterRaw
from humanloop.paths.model_configs_serialize.post import SerializeRaw


class ModelConfigsApiRaw(
    DeserializeRaw,
    ExportRaw,
    GetRaw,
    RegisterRaw,
    SerializeRaw,
):
    """NOTE:
    This class is auto generated by Konfig (https://konfigthis.com)
    """
    pass
