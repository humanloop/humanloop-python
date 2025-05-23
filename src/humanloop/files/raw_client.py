# This file was auto-generated by Fern from our API Definition.

import typing
from json.decoder import JSONDecodeError

from ..core.api_error import ApiError
from ..core.client_wrapper import AsyncClientWrapper, SyncClientWrapper
from ..core.http_response import AsyncHttpResponse, HttpResponse
from ..core.request_options import RequestOptions
from ..core.unchecked_base_model import construct_type
from ..errors.unprocessable_entity_error import UnprocessableEntityError
from ..types.file_sort_by import FileSortBy
from ..types.file_type import FileType
from ..types.http_validation_error import HttpValidationError
from ..types.paginated_data_union_prompt_response_tool_response_dataset_response_evaluator_response_flow_response_agent_response import (
    PaginatedDataUnionPromptResponseToolResponseDatasetResponseEvaluatorResponseFlowResponseAgentResponse,
)
from ..types.sort_order import SortOrder
from .types.retrieve_by_path_files_retrieve_by_path_post_response import RetrieveByPathFilesRetrieveByPathPostResponse

# this is used as the default value for optional parameters
OMIT = typing.cast(typing.Any, ...)


class RawFilesClient:
    def __init__(self, *, client_wrapper: SyncClientWrapper):
        self._client_wrapper = client_wrapper

    def list_files(
        self,
        *,
        page: typing.Optional[int] = None,
        size: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
        path: typing.Optional[str] = None,
        template: typing.Optional[bool] = None,
        type: typing.Optional[typing.Union[FileType, typing.Sequence[FileType]]] = None,
        environment: typing.Optional[str] = None,
        sort_by: typing.Optional[FileSortBy] = None,
        order: typing.Optional[SortOrder] = None,
        include_raw_file_content: typing.Optional[bool] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> HttpResponse[
        PaginatedDataUnionPromptResponseToolResponseDatasetResponseEvaluatorResponseFlowResponseAgentResponse
    ]:
        """
        Get a paginated list of files.

        Parameters
        ----------
        page : typing.Optional[int]
            Page offset for pagination.

        size : typing.Optional[int]
            Page size for pagination. Number of files to fetch.

        name : typing.Optional[str]
            Case-insensitive filter for file name.

        path : typing.Optional[str]
            Path of the directory to filter for. Returns files in this directory and all its subdirectories.

        template : typing.Optional[bool]
            Filter to include only template files.

        type : typing.Optional[typing.Union[FileType, typing.Sequence[FileType]]]
            List of file types to filter for.

        environment : typing.Optional[str]
            Case-sensitive filter for files with a deployment in the specified environment. Requires the environment name.

        sort_by : typing.Optional[FileSortBy]
            Field to sort files by

        order : typing.Optional[SortOrder]
            Direction to sort by.

        include_raw_file_content : typing.Optional[bool]
            Whether to include the raw file content in the response. Currently only supported for Agents and Prompts.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        HttpResponse[PaginatedDataUnionPromptResponseToolResponseDatasetResponseEvaluatorResponseFlowResponseAgentResponse]
            Successful Response
        """
        _response = self._client_wrapper.httpx_client.request(
            "files",
            method="GET",
            params={
                "page": page,
                "size": size,
                "name": name,
                "path": path,
                "template": template,
                "type": type,
                "environment": environment,
                "sort_by": sort_by,
                "order": order,
                "include_raw_file_content": include_raw_file_content,
            },
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                _data = typing.cast(
                    PaginatedDataUnionPromptResponseToolResponseDatasetResponseEvaluatorResponseFlowResponseAgentResponse,
                    construct_type(
                        type_=PaginatedDataUnionPromptResponseToolResponseDatasetResponseEvaluatorResponseFlowResponseAgentResponse,  # type: ignore
                        object_=_response.json(),
                    ),
                )
                return HttpResponse(response=_response, data=_data)
            if _response.status_code == 422:
                raise UnprocessableEntityError(
                    headers=dict(_response.headers),
                    body=typing.cast(
                        HttpValidationError,
                        construct_type(
                            type_=HttpValidationError,  # type: ignore
                            object_=_response.json(),
                        ),
                    ),
                )
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response.text)
        raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response_json)

    def retrieve_by_path(
        self,
        *,
        path: str,
        environment: typing.Optional[str] = None,
        include_raw_file_content: typing.Optional[bool] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> HttpResponse[RetrieveByPathFilesRetrieveByPathPostResponse]:
        """
        Retrieve a File by path.

        Parameters
        ----------
        path : str
            Path of the File to retrieve.

        environment : typing.Optional[str]
            Name of the Environment to retrieve a deployed Version from.

        include_raw_file_content : typing.Optional[bool]
            Whether to include the raw file content in the response. Currently only supported for Agents and Prompts.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        HttpResponse[RetrieveByPathFilesRetrieveByPathPostResponse]
            Successful Response
        """
        _response = self._client_wrapper.httpx_client.request(
            "files/retrieve-by-path",
            method="POST",
            params={
                "environment": environment,
                "include_raw_file_content": include_raw_file_content,
            },
            json={
                "path": path,
            },
            headers={
                "content-type": "application/json",
            },
            request_options=request_options,
            omit=OMIT,
        )
        try:
            if 200 <= _response.status_code < 300:
                _data = typing.cast(
                    RetrieveByPathFilesRetrieveByPathPostResponse,
                    construct_type(
                        type_=RetrieveByPathFilesRetrieveByPathPostResponse,  # type: ignore
                        object_=_response.json(),
                    ),
                )
                return HttpResponse(response=_response, data=_data)
            if _response.status_code == 422:
                raise UnprocessableEntityError(
                    headers=dict(_response.headers),
                    body=typing.cast(
                        HttpValidationError,
                        construct_type(
                            type_=HttpValidationError,  # type: ignore
                            object_=_response.json(),
                        ),
                    ),
                )
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response.text)
        raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response_json)


class AsyncRawFilesClient:
    def __init__(self, *, client_wrapper: AsyncClientWrapper):
        self._client_wrapper = client_wrapper

    async def list_files(
        self,
        *,
        page: typing.Optional[int] = None,
        size: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
        path: typing.Optional[str] = None,
        template: typing.Optional[bool] = None,
        type: typing.Optional[typing.Union[FileType, typing.Sequence[FileType]]] = None,
        environment: typing.Optional[str] = None,
        sort_by: typing.Optional[FileSortBy] = None,
        order: typing.Optional[SortOrder] = None,
        include_raw_file_content: typing.Optional[bool] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> AsyncHttpResponse[
        PaginatedDataUnionPromptResponseToolResponseDatasetResponseEvaluatorResponseFlowResponseAgentResponse
    ]:
        """
        Get a paginated list of files.

        Parameters
        ----------
        page : typing.Optional[int]
            Page offset for pagination.

        size : typing.Optional[int]
            Page size for pagination. Number of files to fetch.

        name : typing.Optional[str]
            Case-insensitive filter for file name.

        path : typing.Optional[str]
            Path of the directory to filter for. Returns files in this directory and all its subdirectories.

        template : typing.Optional[bool]
            Filter to include only template files.

        type : typing.Optional[typing.Union[FileType, typing.Sequence[FileType]]]
            List of file types to filter for.

        environment : typing.Optional[str]
            Case-sensitive filter for files with a deployment in the specified environment. Requires the environment name.

        sort_by : typing.Optional[FileSortBy]
            Field to sort files by

        order : typing.Optional[SortOrder]
            Direction to sort by.

        include_raw_file_content : typing.Optional[bool]
            Whether to include the raw file content in the response. Currently only supported for Agents and Prompts.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        AsyncHttpResponse[PaginatedDataUnionPromptResponseToolResponseDatasetResponseEvaluatorResponseFlowResponseAgentResponse]
            Successful Response
        """
        _response = await self._client_wrapper.httpx_client.request(
            "files",
            method="GET",
            params={
                "page": page,
                "size": size,
                "name": name,
                "path": path,
                "template": template,
                "type": type,
                "environment": environment,
                "sort_by": sort_by,
                "order": order,
                "include_raw_file_content": include_raw_file_content,
            },
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                _data = typing.cast(
                    PaginatedDataUnionPromptResponseToolResponseDatasetResponseEvaluatorResponseFlowResponseAgentResponse,
                    construct_type(
                        type_=PaginatedDataUnionPromptResponseToolResponseDatasetResponseEvaluatorResponseFlowResponseAgentResponse,  # type: ignore
                        object_=_response.json(),
                    ),
                )
                return AsyncHttpResponse(response=_response, data=_data)
            if _response.status_code == 422:
                raise UnprocessableEntityError(
                    headers=dict(_response.headers),
                    body=typing.cast(
                        HttpValidationError,
                        construct_type(
                            type_=HttpValidationError,  # type: ignore
                            object_=_response.json(),
                        ),
                    ),
                )
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response.text)
        raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response_json)

    async def retrieve_by_path(
        self,
        *,
        path: str,
        environment: typing.Optional[str] = None,
        include_raw_file_content: typing.Optional[bool] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> AsyncHttpResponse[RetrieveByPathFilesRetrieveByPathPostResponse]:
        """
        Retrieve a File by path.

        Parameters
        ----------
        path : str
            Path of the File to retrieve.

        environment : typing.Optional[str]
            Name of the Environment to retrieve a deployed Version from.

        include_raw_file_content : typing.Optional[bool]
            Whether to include the raw file content in the response. Currently only supported for Agents and Prompts.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        AsyncHttpResponse[RetrieveByPathFilesRetrieveByPathPostResponse]
            Successful Response
        """
        _response = await self._client_wrapper.httpx_client.request(
            "files/retrieve-by-path",
            method="POST",
            params={
                "environment": environment,
                "include_raw_file_content": include_raw_file_content,
            },
            json={
                "path": path,
            },
            headers={
                "content-type": "application/json",
            },
            request_options=request_options,
            omit=OMIT,
        )
        try:
            if 200 <= _response.status_code < 300:
                _data = typing.cast(
                    RetrieveByPathFilesRetrieveByPathPostResponse,
                    construct_type(
                        type_=RetrieveByPathFilesRetrieveByPathPostResponse,  # type: ignore
                        object_=_response.json(),
                    ),
                )
                return AsyncHttpResponse(response=_response, data=_data)
            if _response.status_code == 422:
                raise UnprocessableEntityError(
                    headers=dict(_response.headers),
                    body=typing.cast(
                        HttpValidationError,
                        construct_type(
                            type_=HttpValidationError,  # type: ignore
                            object_=_response.json(),
                        ),
                    ),
                )
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response.text)
        raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response_json)
