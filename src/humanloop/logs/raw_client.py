# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing
from json.decoder import JSONDecodeError

from ..core.api_error import ApiError
from ..core.client_wrapper import AsyncClientWrapper, SyncClientWrapper
from ..core.datetime_utils import serialize_datetime
from ..core.http_response import AsyncHttpResponse, HttpResponse
from ..core.jsonable_encoder import jsonable_encoder
from ..core.pagination import AsyncPager, BaseHttpResponse, SyncPager
from ..core.request_options import RequestOptions
from ..core.unchecked_base_model import construct_type
from ..errors.unprocessable_entity_error import UnprocessableEntityError
from ..types.http_validation_error import HttpValidationError
from ..types.log_response import LogResponse
from ..types.paginated_data_log_response import PaginatedDataLogResponse


class RawLogsClient:
    def __init__(self, *, client_wrapper: SyncClientWrapper):
        self._client_wrapper = client_wrapper

    def list(
        self,
        *,
        file_id: str,
        page: typing.Optional[int] = None,
        size: typing.Optional[int] = None,
        version_id: typing.Optional[str] = None,
        id: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
        search: typing.Optional[str] = None,
        metadata_search: typing.Optional[str] = None,
        start_date: typing.Optional[dt.datetime] = None,
        end_date: typing.Optional[dt.datetime] = None,
        include_parent: typing.Optional[bool] = None,
        in_trace_filter: typing.Optional[typing.Union[bool, typing.Sequence[bool]]] = None,
        sample: typing.Optional[int] = None,
        include_trace_children: typing.Optional[bool] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> SyncPager[LogResponse]:
        """
        List all Logs for the given filter criteria.

        Parameters
        ----------
        file_id : str
            Unique identifier for the File to list Logs for.

        page : typing.Optional[int]
            Page number for pagination.

        size : typing.Optional[int]
            Page size for pagination. Number of Logs to fetch.

        version_id : typing.Optional[str]
            If provided, only Logs belonging to the specified Version will be returned.

        id : typing.Optional[typing.Union[str, typing.Sequence[str]]]
            If provided, returns Logs whose IDs contain any of the specified values as substrings.

        search : typing.Optional[str]
            If provided, only Logs that contain the provided string in its inputs and output will be returned.

        metadata_search : typing.Optional[str]
            If provided, only Logs that contain the provided string in its metadata will be returned.

        start_date : typing.Optional[dt.datetime]
            If provided, only Logs created after the specified date will be returned.

        end_date : typing.Optional[dt.datetime]
            If provided, only Logs created before the specified date will be returned.

        include_parent : typing.Optional[bool]
            If true, include the full parent Log in the response. Only applicable when retrieving Evaluator Logs.

        in_trace_filter : typing.Optional[typing.Union[bool, typing.Sequence[bool]]]
            If true, return Logs that are associated to a Trace. False, return Logs that are not associated to a Trace.

        sample : typing.Optional[int]
            If provided, limit the response to a random subset of logs from the filtered results. (This will be an approximate sample, not a strict limit.)

        include_trace_children : typing.Optional[bool]
            If true, populate `trace_children` for the retrieved Logs. Only applicable when retrieving Flow or Agent Logs.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        SyncPager[LogResponse]
            Successful Response
        """
        page = page if page is not None else 1

        _response = self._client_wrapper.httpx_client.request(
            "logs",
            method="GET",
            params={
                "file_id": file_id,
                "page": page,
                "size": size,
                "version_id": version_id,
                "id": id,
                "search": search,
                "metadata_search": metadata_search,
                "start_date": serialize_datetime(start_date) if start_date is not None else None,
                "end_date": serialize_datetime(end_date) if end_date is not None else None,
                "include_parent": include_parent,
                "in_trace_filter": in_trace_filter,
                "sample": sample,
                "include_trace_children": include_trace_children,
            },
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                _parsed_response = typing.cast(
                    PaginatedDataLogResponse,
                    construct_type(
                        type_=PaginatedDataLogResponse,  # type: ignore
                        object_=_response.json(),
                    ),
                )
                _items = _parsed_response.records
                _has_next = True
                _get_next = lambda: self.list(
                    file_id=file_id,
                    page=page + 1,
                    size=size,
                    version_id=version_id,
                    id=id,
                    search=search,
                    metadata_search=metadata_search,
                    start_date=start_date,
                    end_date=end_date,
                    include_parent=include_parent,
                    in_trace_filter=in_trace_filter,
                    sample=sample,
                    include_trace_children=include_trace_children,
                    request_options=request_options,
                )
                return SyncPager(
                    has_next=_has_next, items=_items, get_next=_get_next, response=BaseHttpResponse(response=_response)
                )
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

    def delete(
        self,
        *,
        id: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> HttpResponse[None]:
        """
        Delete Logs with the given IDs.

        Parameters
        ----------
        id : typing.Optional[typing.Union[str, typing.Sequence[str]]]
            Unique identifiers for the Logs to delete.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        HttpResponse[None]
        """
        _response = self._client_wrapper.httpx_client.request(
            "logs",
            method="DELETE",
            params={
                "id": id,
            },
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return HttpResponse(response=_response, data=None)
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

    def get(self, id: str, *, request_options: typing.Optional[RequestOptions] = None) -> HttpResponse[LogResponse]:
        """
        Retrieve the Log with the given ID.

        Parameters
        ----------
        id : str
            Unique identifier for Log.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        HttpResponse[LogResponse]
            Successful Response
        """
        _response = self._client_wrapper.httpx_client.request(
            f"logs/{jsonable_encoder(id)}",
            method="GET",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                _data = typing.cast(
                    LogResponse,
                    construct_type(
                        type_=LogResponse,  # type: ignore
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


class AsyncRawLogsClient:
    def __init__(self, *, client_wrapper: AsyncClientWrapper):
        self._client_wrapper = client_wrapper

    async def list(
        self,
        *,
        file_id: str,
        page: typing.Optional[int] = None,
        size: typing.Optional[int] = None,
        version_id: typing.Optional[str] = None,
        id: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
        search: typing.Optional[str] = None,
        metadata_search: typing.Optional[str] = None,
        start_date: typing.Optional[dt.datetime] = None,
        end_date: typing.Optional[dt.datetime] = None,
        include_parent: typing.Optional[bool] = None,
        in_trace_filter: typing.Optional[typing.Union[bool, typing.Sequence[bool]]] = None,
        sample: typing.Optional[int] = None,
        include_trace_children: typing.Optional[bool] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> AsyncPager[LogResponse]:
        """
        List all Logs for the given filter criteria.

        Parameters
        ----------
        file_id : str
            Unique identifier for the File to list Logs for.

        page : typing.Optional[int]
            Page number for pagination.

        size : typing.Optional[int]
            Page size for pagination. Number of Logs to fetch.

        version_id : typing.Optional[str]
            If provided, only Logs belonging to the specified Version will be returned.

        id : typing.Optional[typing.Union[str, typing.Sequence[str]]]
            If provided, returns Logs whose IDs contain any of the specified values as substrings.

        search : typing.Optional[str]
            If provided, only Logs that contain the provided string in its inputs and output will be returned.

        metadata_search : typing.Optional[str]
            If provided, only Logs that contain the provided string in its metadata will be returned.

        start_date : typing.Optional[dt.datetime]
            If provided, only Logs created after the specified date will be returned.

        end_date : typing.Optional[dt.datetime]
            If provided, only Logs created before the specified date will be returned.

        include_parent : typing.Optional[bool]
            If true, include the full parent Log in the response. Only applicable when retrieving Evaluator Logs.

        in_trace_filter : typing.Optional[typing.Union[bool, typing.Sequence[bool]]]
            If true, return Logs that are associated to a Trace. False, return Logs that are not associated to a Trace.

        sample : typing.Optional[int]
            If provided, limit the response to a random subset of logs from the filtered results. (This will be an approximate sample, not a strict limit.)

        include_trace_children : typing.Optional[bool]
            If true, populate `trace_children` for the retrieved Logs. Only applicable when retrieving Flow or Agent Logs.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        AsyncPager[LogResponse]
            Successful Response
        """
        page = page if page is not None else 1

        _response = await self._client_wrapper.httpx_client.request(
            "logs",
            method="GET",
            params={
                "file_id": file_id,
                "page": page,
                "size": size,
                "version_id": version_id,
                "id": id,
                "search": search,
                "metadata_search": metadata_search,
                "start_date": serialize_datetime(start_date) if start_date is not None else None,
                "end_date": serialize_datetime(end_date) if end_date is not None else None,
                "include_parent": include_parent,
                "in_trace_filter": in_trace_filter,
                "sample": sample,
                "include_trace_children": include_trace_children,
            },
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                _parsed_response = typing.cast(
                    PaginatedDataLogResponse,
                    construct_type(
                        type_=PaginatedDataLogResponse,  # type: ignore
                        object_=_response.json(),
                    ),
                )
                _items = _parsed_response.records
                _has_next = True

                async def _get_next():
                    return await self.list(
                        file_id=file_id,
                        page=page + 1,
                        size=size,
                        version_id=version_id,
                        id=id,
                        search=search,
                        metadata_search=metadata_search,
                        start_date=start_date,
                        end_date=end_date,
                        include_parent=include_parent,
                        in_trace_filter=in_trace_filter,
                        sample=sample,
                        include_trace_children=include_trace_children,
                        request_options=request_options,
                    )

                return AsyncPager(
                    has_next=_has_next, items=_items, get_next=_get_next, response=BaseHttpResponse(response=_response)
                )
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

    async def delete(
        self,
        *,
        id: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> AsyncHttpResponse[None]:
        """
        Delete Logs with the given IDs.

        Parameters
        ----------
        id : typing.Optional[typing.Union[str, typing.Sequence[str]]]
            Unique identifiers for the Logs to delete.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        AsyncHttpResponse[None]
        """
        _response = await self._client_wrapper.httpx_client.request(
            "logs",
            method="DELETE",
            params={
                "id": id,
            },
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return AsyncHttpResponse(response=_response, data=None)
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

    async def get(
        self, id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> AsyncHttpResponse[LogResponse]:
        """
        Retrieve the Log with the given ID.

        Parameters
        ----------
        id : str
            Unique identifier for Log.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        AsyncHttpResponse[LogResponse]
            Successful Response
        """
        _response = await self._client_wrapper.httpx_client.request(
            f"logs/{jsonable_encoder(id)}",
            method="GET",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                _data = typing.cast(
                    LogResponse,
                    construct_type(
                        type_=LogResponse,  # type: ignore
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
