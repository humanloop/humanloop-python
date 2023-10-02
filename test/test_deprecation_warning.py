import unittest
from unittest.mock import patch

from humanloop.api_client import ApiClient, DeprecationWarningOnce
from humanloop.configuration import Configuration


class RandomClass:
    configuration = Configuration(
        # Defining the host is optional and defaults to https://api.humanloop.com/v4
        # See configuration.py for a list of all supported configuration parameters.
        host = "https://api.humanloop.com/v4",
        openai_api_key = 'OPENAI_API_KEY',
        openai_azure_api_key = 'OPENAI_AZURE_API_KEY',
        openai_azure_endpoint_api_key = 'OPENAI_AZURE_ENDPOINT_API_KEY',
        ai21_api_key = 'AI21_API_KEY',
        mock_api_key = 'MOCK_API_KEY',
        anthropic_api_key = 'ANTHROPIC_API_KEY',
        cohere_api_key = 'COHERE_API_KEY',
    
        # Configure API key authorization: APIKeyHeader
        api_key = 'YOUR_API_KEY',
        # Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
        # api_key_prefix = {'APIKeyHeader': 'Bearer'},
    )
    api_client = ApiClient(configuration)

    @DeprecationWarningOnce
    def deprecated_method(self):
        return "Method called"

    @DeprecationWarningOnce(prefix="tag")
    def deprecated_method_with_prefix(self):
        return "Method with prefix called"


class TestDeprecationWarning(unittest.TestCase):
    @patch("logging.Logger.warning")
    def test_deprecation_warning_without_prefix(self, mock_warning):
        obj = RandomClass()

        obj.deprecated_method()
        obj.deprecated_method()

        # Check that the logger.warning() was called once
        self.assertEqual(mock_warning.call_count, 1)

        # Get the warning message
        warning_msg = mock_warning.call_args[0][0]

        # Check the content of the warning message
        self.assertNotIn("tag", warning_msg)

    @patch("logging.Logger.warning")
    def test_deprecation_warning_with_prefix(self, mock_warning):
        obj = RandomClass()

        obj.deprecated_method_with_prefix()
        obj.deprecated_method_with_prefix()

        # Check that the logger.warning() was called once
        self.assertEqual(mock_warning.call_count, 1)

        # Get the warning message
        warning_msg = mock_warning.call_args[0][0]

        # Check the content of the warning message
        self.assertIn("tag", warning_msg)
