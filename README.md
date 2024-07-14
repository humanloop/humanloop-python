# Humanloop Python Library

[![fern shield](https://img.shields.io/badge/%F0%9F%8C%BF-SDK%20generated%20by%20Fern-brightgreen)](https://github.com/fern-api/fern)
[![pypi](https://img.shields.io/pypi/v/humanloop)](https://pypi.python.org/pypi/humanloop)

The Humanloop Python library provides convenient access to the Humanloop API from Python.

## Installation

```sh
pip install humanloop
```

## Usage

Instantiate and use the client with the following:

```python
from humanloop import ChatMessage, PromptKernelRequest
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)
client.prompts.log(
    path="persona",
    prompt=PromptKernelRequest(
        model="gpt-4",
        template=[
            ChatMessage(
                role="system",
                content="You are {{person}}. Answer questions as this person. Do not break character.",
            )
        ],
    ),
    messages=[
        ChatMessage(
            role="user",
            content="What really happened at Roswell?",
        )
    ],
    inputs={"person": "Trump"},
)
```

## Async Client

The SDK also exports an `async` client so that you can make non-blocking calls to our API.

```python
from humanloop import ChatMessage, PromptKernelRequest
from humanloop.client import AsyncHumanloop

client = AsyncHumanloop(
    api_key="YOUR_API_KEY",
)
await client.prompts.log(
    path="persona",
    prompt=PromptKernelRequest(
        model="gpt-4",
        template=[
            ChatMessage(
                role="system",
                content="You are {{person}}. Answer questions as this person. Do not break character.",
            )
        ],
    ),
    messages=[
        ChatMessage(
            role="user",
            content="What really happened at Roswell?",
        )
    ],
    inputs={"person": "Trump"},
)
```

## Exception Handling

All errors thrown by the SDK will be subclasses of [`ApiError`](./src/schematic/core/api_error.py).

```python
import humanloop

try:
    client.prompts.call(...)
except humanloop.core.ApiError as e: # Handle all errors
  print(e.status_code)
  print(e.body)
```

## Advanced

### Timeouts

By default, requests time out after 60 seconds. You can configure this with a
timeout option at the client or request level.

```python
from humanloop.client import Humanloop

client = Humanloop(
    ...,
    # All timeouts are 20 seconds
    timeout=20.0,
)

# Override timeout for a specific method
client.prompts.call(..., {
    timeout_in_seconds=20.0
})
```

### Retries

The SDK is instrumented with automatic retries with exponential backoff. A request will be
retried as long as the request is deemed retriable and the number of retry attempts has not grown larger
than the configured retry limit (default: 2).

A request is deemed retriable when any of the following HTTP status codes is returned:

- [408](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/408) (Timeout)
- [429](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429) (Too Many Requests)
- [5XX](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500) (Internal Server Errors)

Use the `max_retries` request option to configure this behavior.

```python
client.prompts.call(..., {
     max_retries=1
})
```

### Custom HTTP client

You can override the httpx client to customize it for your use-case. Some common use-cases
include support for proxies and transports.

```python
import httpx

from humanloop.client import Humanloop

client = Humanloop(...,
    http_client=httpx.Client(
        proxies="http://my.test.proxy.example.com",
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    ),
)
```

## Pagination

Paginated requests will return a `SyncPager` or `AsyncPager`, which can be used as generators for the underlying object.

```python
from humanloop.client import Humanloop

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

## Contributing

While we value open-source contributions to this SDK, this library is generated programmatically.
Additions made directly to this library would have to be moved over to our generation code,
otherwise they would be overwritten upon the next generated release. Feel free to open a PR as
a proof of concept, but know that we will not be able to merge it as-is. We suggest opening
an issue first to discuss with us!

On the other hand, contributions to the README are always very welcome!
