from opentelemetry.proto.common.v1.common_pb2 import KeyValue, AnyValue, InstrumentationScope
from opentelemetry.proto.trace.v1.trace_pb2 import (
    TracesData,
    ResourceSpans,
    ScopeSpans,
    Span as ProtoBufferSpan,
)
from google.protobuf.json_format import MessageToJson

from opentelemetry.sdk.trace import ReadableSpan
from humanloop.otel.helpers import is_llm_provider_call


def serialize_span(span_to_export: ReadableSpan) -> str:
    """
    Serialize a span into format compatible with the /otel backend endpoint.
    """
    payload = TracesData(
        resource_spans=[
            ResourceSpans(
                scope_spans=[
                    ScopeSpans(
                        scope=InstrumentationScope(
                            name="humanloop.sdk.provider"
                            if is_llm_provider_call(span_to_export)
                            else "humanloop.sdk.decorator",
                            version="0.1.0",
                        ),
                        spans=[
                            ProtoBufferSpan(
                                trace_id=span_to_export.context.trace_id.to_bytes(length=16, byteorder="big"),
                                span_id=span_to_export.context.span_id.to_bytes(length=8, byteorder="big"),
                                name=span_to_export.name,
                                kind={
                                    0: ProtoBufferSpan.SpanKind.SPAN_KIND_INTERNAL,
                                    1: ProtoBufferSpan.SpanKind.SPAN_KIND_SERVER,
                                    2: ProtoBufferSpan.SpanKind.SPAN_KIND_CLIENT,
                                    3: ProtoBufferSpan.SpanKind.SPAN_KIND_PRODUCER,
                                    4: ProtoBufferSpan.SpanKind.SPAN_KIND_CONSUMER,
                                }[span_to_export.kind.value],
                                start_time_unix_nano=span_to_export.start_time,  # type: ignore [attr-defined, arg-type]
                                end_time_unix_nano=span_to_export.end_time,  # type: ignore [attr-defined, arg-type]
                                attributes=[
                                    KeyValue(
                                        key=key,
                                        value=AnyValue(string_value=str(value)),
                                    )
                                    for key, value in span_to_export.attributes.items()  # type: ignore [union-attr]
                                ],
                                dropped_attributes_count=span_to_export.dropped_attributes,
                                dropped_events_count=span_to_export.dropped_events,
                                dropped_links_count=span_to_export.dropped_links,
                                links=[
                                    ProtoBufferSpan.Link(
                                        trace_id=link.context.trace_id.to_bytes(length=16, byteorder="big"),
                                        span_id=link.context.span_id.to_bytes(length=8, byteorder="big"),
                                        attributes=[
                                            KeyValue(
                                                key=key,
                                                value=AnyValue(string_value=str(value)),
                                            )
                                            for key, value in link.attributes.items()  # type: ignore [union-attr]
                                        ],
                                    )
                                    for link in span_to_export.links
                                ],
                                events=[],
                            )
                        ],
                    )
                ]
            )
        ]
    )

    return MessageToJson(payload)
