# Humanloop Python Library

[![fern shield](https://img.shields.io/badge/%F0%9F%8C%BF-SDK%20generated%20by%20Fern-brightgreen)](https://github.com/fern-api/fern)

The Humanloop Python Library provides convenient access to the Humanloop API from
applications written in Python.

The library includes type definitions for all
request and response fields, and offers both synchronous and asynchronous clients powered by httpx.

## Installation

Add this dependency to your project's build file:

```bash
pip install humanloop
# or
poetry add humanloop
```

## Usage

Simply import `Humanloop` and start making calls to our API.

```python
from humanloop import ChatMessage
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY", # Defaults to HUMANLOOP_API_KEY
)

client.prompts.call(
    prompt_id="prompt_id",
    messages=[
        ChatMessage(
            content="What is the day today?",
            role="user",
        )
    ],
)
```

### Typing

To construct payloads you can either use the dedicated types like `ChatMessage` or construct directly from a dictionary like so:

```python
from humanloop import ChatMessage
from humanloop.client import Humanloop

client.prompts.call(
    prompt_id="prompt_id",
    messages=[
        {
            content="Tell me a funny joke",
            role="user",
        }
    ],
)
```

### Streaming

The SDK supports streaming endpoints. To take advantage of this feature for `prompts.call`, simply 
pass in `stream=True` in the request. The response will be a generator that you can loop over.

```Python
from humanloop import ChatMessage
from humanloop.client import Humanloop

client = Humanloop(
    api_key="YOUR_API_KEY",
)

stream = client.prompts.call(
    prompt_id="prompt_id",
    stream=True,
    messages=[
        ChatMessage(
            content="Tell me a funny joke",
            role="user",
        )
    ],
)

for message in stream:
    print(message)
```

### Pagination

Paginated requests will return a `SyncPager` or `AsyncPager`, which can be used as generators for the underlying object. For example, `evaluations.list` will return a generator over `EvaluationResponse` and handle the pagination behind the scenes:

```python
import humanloop.client

client = HumanloopClient(
    api_key="YOUR_API_KEY",
)

for evaluation in client.evaluations.list(file_id="id"):
  print(evaluation)
```

you could also iterate page-by-page:

```python
for page in client.evaluations.list(file_id="id").iter_pages():
  print(page.items)
```

or manually:

```python
pager = client.evaluations.list(project_id="id")
# First page
print(pager.items)
# Second page
pager = pager.next_page()
print(pager.items)
```

## Async Client

The SDK also exports an async client so that you can make non-blocking
calls to our API.

```python
import asyncio
from humanloop import ChatMessage
from humanloop.client import AsyncHumanloop

client = AsyncHumanloop(
    api_key="YOUR_API_KEY",
)

async def main() -> None:
    await client.prompts.call(
        prompt_id="prompt_id",
        messages=[
            ChatMessage(
                content="What is the day today?",
                role="user",
            )
        ],
    )
asyncio.run(main())
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

## Contributing

While we value open-source contributions to this SDK, this library is generated programmatically.
Additions made directly to this library would have to be moved over to our generation code,
otherwise they would be overwritten upon the next generated release. Feel free to open a PR as
a proof of concept, but know that we will not be able to merge it as-is. We suggest opening
an issue first to discuss with us!

On the other hand, contributions to the README are always very welcome!
