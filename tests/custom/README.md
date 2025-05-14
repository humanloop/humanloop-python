# Custom Tests Directory

This directory contains custom tests for the Humanloop Python SDK. While the main SDK is auto-generated using [Fern](https://buildwithfern.com/), this directory allows us to add our own test implementations that won't be overwritten during regeneration.

## Why Custom Tests?

- **Preservation**: Tests in this directory won't be overwritten when regenerating the SDK
- **Custom Implementation**: Allows testing of our own implementations beyond the auto-generated code
- **Integration**: Enables testing of how our custom code works with the auto-generated SDK

## Running Tests

```bash
# Run all custom tests
pytest tests/custom/

# Run specific test file
pytest tests/custom/sync/test_sync_client.py
```
