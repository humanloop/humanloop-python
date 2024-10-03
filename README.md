# Humanloop Python Library

[![fern shield](https://img.shields.io/badge/%F0%9F%8C%BF-Built%20with%20Fern-brightgreen)](https://buildwithfern.com?utm_source=github&utm_medium=github&utm_campaign=readme&utm_source=https%3A%2F%2Fgithub.com%2Fhumanloop%2Fhumanloop-python)
[![pypi](https://img.shields.io/pypi/v/humanloop)](https://pypi.python.org/pypi/humanloop)

The Humanloop Python library provides convenient access to the Humanloop API from Python.

## Installation

```sh
pip install humanloop
```

## Reference

A full reference for this library is available [here](./reference.md).

## Usage

Instantiate and use the client with the following:

```python
import datetime

from humanloop import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.prompts.log(
    path="persona",
    prompt={
        "model": "gpt-4",
        "template": [
            {
                "role": "system",
                "content": "You are {{person}}. Answer questions as this person. Do not break character.",
            }
        ],
    },
    messages=[{"role": "user", "content": "What really happened at Roswell?"}],
    inputs={"person": "Trump"},
    created_at=datetime.datetime.fromisoformat(
        "2024-07-19 00:29:35.178000+00:00",
    ),
    provider_latency=6.5931549072265625,
    output_message={
        "content": "Well, you know, there is so much secrecy involved in government, folks, it's unbelievable. They don't want to tell you everything. They don't tell me everything! But about Roswell, it’s a very popular question. I know, I just know, that something very, very peculiar happened there. Was it a weather balloon? Maybe. Was it something extraterrestrial? Could be. I'd love to go down and open up all the classified documents, believe me, I would. But they don't let that happen. The Deep State, folks, the Deep State. They’re unbelievable. They want to keep everything a secret. But whatever the truth is, I can tell you this: it’s something big, very very big. Tremendous, in fact.",
        "role": "assistant",
    },
    prompt_tokens=100,
    output_tokens=220,
    prompt_cost=1e-05,
    output_cost=0.0002,
    finish_reason="stop",
)
```

## Async Client

The SDK also exports an `async` client so that you can make non-blocking calls to our API.

```python
import asyncio
import datetime

from humanloop import AsyncHumanloop

client = AsyncHumanloop(
    api_key="YOUR_API_KEY",
)


async def main() -> None:
    await client.prompts.log(
        path="persona",
        prompt={
            "model": "gpt-4",
            "template": [
                {
                    "role": "system",
                    "content": "You are {{person}}. Answer questions as this person. Do not break character.",
                }
            ],
        },
        messages=[
            {"role": "user", "content": "What really happened at Roswell?"}
        ],
        inputs={"person": "Trump"},
        created_at=datetime.datetime.fromisoformat(
            "2024-07-19 00:29:35.178000+00:00",
        ),
        provider_latency=6.5931549072265625,
        output_message={
            "content": "Well, you know, there is so much secrecy involved in government, folks, it's unbelievable. They don't want to tell you everything. They don't tell me everything! But about Roswell, it’s a very popular question. I know, I just know, that something very, very peculiar happened there. Was it a weather balloon? Maybe. Was it something extraterrestrial? Could be. I'd love to go down and open up all the classified documents, believe me, I would. But they don't let that happen. The Deep State, folks, the Deep State. They’re unbelievable. They want to keep everything a secret. But whatever the truth is, I can tell you this: it’s something big, very very big. Tremendous, in fact.",
            "role": "assistant",
        },
        prompt_tokens=100,
        output_tokens=220,
        prompt_cost=1e-05,
        output_cost=0.0002,
        finish_reason="stop",
    )


asyncio.run(main())
```

## Exception Handling

When the API returns a non-success status code (4xx or 5xx response), a subclass of the following error
will be thrown.

```python
from humanloop.core.api_error import ApiError

try:
    client.prompts.log(...)
except ApiError as e:
    print(e.status_code)
    print(e.body)
```

## Streaming

The SDK supports streaming responses, as well, the response will be a generator that you can loop over.

```python
import datetime

from humanloop import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
response = client.prompts.call_stream(
    version_id="string",
    environment="string",
    path="string",
    id="string",
    messages=[
        {
            "content": "string",
            "name": "string",
            "tool_call_id": "string",
            "role": "user",
            "tool_calls": [
                {
                    "id": "string",
                    "type": "function",
                    "function": {"name": "string"},
                }
            ],
        }
    ],
    prompt={"model": "string"},
    inputs={"string": {"key": "value"}},
    source="string",
    metadata={"string": {"key": "value"}},
    start_time=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
    end_time=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
    source_datapoint_id="string",
    trace_parent_id="string",
    batch_id="string",
    user="string",
    prompts_call_stream_request_environment="string",
    save=True,
    provider_api_keys={
        "openai": "string",
        "ai_21": "string",
        "mock": "string",
        "anthropic": "string",
        "bedrock": "string",
        "cohere": "string",
        "openai_azure": "string",
        "openai_azure_endpoint": "string",
    },
    num_samples=1,
    return_inputs=True,
    logprobs=1,
    suffix="string",
)
for chunk in response:
    yield chunk
```

## Pagination

Paginated requests will return a `SyncPager` or `AsyncPager`, which can be used as generators for the underlying object.

```python
from humanloop import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
response = client.prompts.list(
    size=1,
)
for item in response:
    yield item
# alternatively, you can paginate page-by-page
for page in response.iter_pages():
    yield page
```

## Advanced

### Retries

The SDK is instrumented with automatic retries with exponential backoff. A request will be retried as long
as the request is deemed retriable and the number of retry attempts has not grown larger than the configured
retry limit (default: 2).

A request is deemed retriable when any of the following HTTP status codes is returned:

- [408](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/408) (Timeout)
- [429](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429) (Too Many Requests)
- [5XX](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500) (Internal Server Errors)

Use the `max_retries` request option to configure this behavior.

```python
client.prompts.log(..., request_options={
    "max_retries": 1
})
```

### Timeouts

The SDK defaults to a 60 second timeout. You can configure this with a timeout option at the client or request level.

```python

from humanloop import Humanloop

client = Humanloop(
    ...,
    timeout=20.0,
)


# Override timeout for a specific method
client.prompts.log(..., request_options={
    "timeout_in_seconds": 1
})
```

### Custom Client

You can override the `httpx` client to customize it for your use-case. Some common use-cases include support for proxies
and transports.
```python
import httpx
from humanloop import Humanloop

client = Humanloop(
    ...,
    httpx_client=httpx.Client(
        proxies="http://my.test.proxy.example.com",
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    ),
)
```

## Contributing

While we value open-source contributions to this SDK, this library is generated programmatically.
Additions made directly to this library would have to be moved over to our generation code,
otherwise they would be overwritten upon the next generated release. Feel free to open a PR as
a proof of concept, but know that we will not be able to merge it as-is. We suggest opening
an issue first to discuss with us!

On the other hand, contributions to the README are always very welcome!
