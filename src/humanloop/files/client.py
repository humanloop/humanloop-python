# This file was auto-generated by Fern from our API Definition.

import typing
from ..core.client_wrapper import SyncClientWrapper
from .raw_client import RawFilesClient
from ..types.file_type import FileType
from ..types.project_sort_by import ProjectSortBy
from ..types.sort_order import SortOrder
from ..core.request_options import RequestOptions
from ..types.paginated_data_union_prompt_response_tool_response_dataset_response_evaluator_response_flow_response import (
    PaginatedDataUnionPromptResponseToolResponseDatasetResponseEvaluatorResponseFlowResponse,
)
from .types.retrieve_by_path_files_retrieve_by_path_post_response import RetrieveByPathFilesRetrieveByPathPostResponse
from ..core.client_wrapper import AsyncClientWrapper
from .raw_client import AsyncRawFilesClient

# this is used as the default value for optional parameters
OMIT = typing.cast(typing.Any, ...)


class FilesClient:
    def __init__(self, *, client_wrapper: SyncClientWrapper):
        self._raw_client = RawFilesClient(client_wrapper=client_wrapper)

    @property
    def with_raw_response(self) -> RawFilesClient:
        """
        Retrieves a raw implementation of this client that returns raw responses.

        Returns
        -------
        RawFilesClient
        """
        return self._raw_client

    def list_files(
        self,
        *,
        page: typing.Optional[int] = None,
        size: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
        template: typing.Optional[bool] = None,
        type: typing.Optional[typing.Union[FileType, typing.Sequence[FileType]]] = None,
        environment: typing.Optional[str] = None,
        sort_by: typing.Optional[ProjectSortBy] = None,
        order: typing.Optional[SortOrder] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> PaginatedDataUnionPromptResponseToolResponseDatasetResponseEvaluatorResponseFlowResponse:
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

        template : typing.Optional[bool]
            Filter to include only template files.

        type : typing.Optional[typing.Union[FileType, typing.Sequence[FileType]]]
            List of file types to filter for.

        environment : typing.Optional[str]
            Case-sensitive filter for files with a deployment in the specified environment. Requires the environment name.

        sort_by : typing.Optional[ProjectSortBy]
            Field to sort files by

        order : typing.Optional[SortOrder]
            Direction to sort by.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        PaginatedDataUnionPromptResponseToolResponseDatasetResponseEvaluatorResponseFlowResponse
            Successful Response

        Examples
        --------
        from humanloop import Humanloop

        client = Humanloop(
            api_key="YOUR_API_KEY",
        )
        client.files.list_files()
        """
        response = self._raw_client.list_files(
            page=page,
            size=size,
            name=name,
            template=template,
            type=type,
            environment=environment,
            sort_by=sort_by,
            order=order,
            request_options=request_options,
        )
        return response.data

    def retrieve_by_path(
        self,
        *,
        path: str,
        environment: typing.Optional[str] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> RetrieveByPathFilesRetrieveByPathPostResponse:
        """
        Retrieve a File by path.

        Parameters
        ----------
        path : str
            Path of the File to retrieve.

        environment : typing.Optional[str]
            Name of the Environment to retrieve a deployed Version from.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        RetrieveByPathFilesRetrieveByPathPostResponse
            Successful Response

        Examples
        --------
        from humanloop import Humanloop

        client = Humanloop(
            api_key="YOUR_API_KEY",
        )
        client.files.retrieve_by_path(
            path="path",
        )
        """
        response = self._raw_client.retrieve_by_path(
            path=path, environment=environment, request_options=request_options
        )
        return response.data


class AsyncFilesClient:
    def __init__(self, *, client_wrapper: AsyncClientWrapper):
        self._raw_client = AsyncRawFilesClient(client_wrapper=client_wrapper)

    @property
    def with_raw_response(self) -> AsyncRawFilesClient:
        """
        Retrieves a raw implementation of this client that returns raw responses.

        Returns
        -------
        AsyncRawFilesClient
        """
        return self._raw_client

    async def list_files(
        self,
        *,
        page: typing.Optional[int] = None,
        size: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
        template: typing.Optional[bool] = None,
        type: typing.Optional[typing.Union[FileType, typing.Sequence[FileType]]] = None,
        environment: typing.Optional[str] = None,
        sort_by: typing.Optional[ProjectSortBy] = None,
        order: typing.Optional[SortOrder] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> PaginatedDataUnionPromptResponseToolResponseDatasetResponseEvaluatorResponseFlowResponse:
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

        template : typing.Optional[bool]
            Filter to include only template files.

        type : typing.Optional[typing.Union[FileType, typing.Sequence[FileType]]]
            List of file types to filter for.

        environment : typing.Optional[str]
            Case-sensitive filter for files with a deployment in the specified environment. Requires the environment name.

        sort_by : typing.Optional[ProjectSortBy]
            Field to sort files by

        order : typing.Optional[SortOrder]
            Direction to sort by.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        PaginatedDataUnionPromptResponseToolResponseDatasetResponseEvaluatorResponseFlowResponse
            Successful Response

        Examples
        --------
        import asyncio

        from humanloop import AsyncHumanloop

        client = AsyncHumanloop(
            api_key="YOUR_API_KEY",
        )


        async def main() -> None:
            await client.files.list_files()


        asyncio.run(main())
        """
        response = await self._raw_client.list_files(
            page=page,
            size=size,
            name=name,
            template=template,
            type=type,
            environment=environment,
            sort_by=sort_by,
            order=order,
            request_options=request_options,
        )
        return response.data

    async def retrieve_by_path(
        self,
        *,
        path: str,
        environment: typing.Optional[str] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> RetrieveByPathFilesRetrieveByPathPostResponse:
        """
        Retrieve a File by path.

        Parameters
        ----------
        path : str
            Path of the File to retrieve.

        environment : typing.Optional[str]
            Name of the Environment to retrieve a deployed Version from.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        RetrieveByPathFilesRetrieveByPathPostResponse
            Successful Response

        Examples
        --------
        import asyncio

        from humanloop import AsyncHumanloop

        client = AsyncHumanloop(
            api_key="YOUR_API_KEY",
        )


        async def main() -> None:
            await client.files.retrieve_by_path(
                path="path",
            )


        asyncio.run(main())
        """
        response = await self._raw_client.retrieve_by_path(
            path=path, environment=environment, request_options=request_options
        )
        return response.data
