from typing_extensions import NotRequired

from humanloop.requests.prompt_kernel_request import PromptKernelRequestParams


class DecoratorPromptKernelRequestParams(PromptKernelRequestParams):
    """See :class:`PromptKernelRequestParams` for more information.

    Allows the `model` field to be optional for Prompt decorator.
    """

    model: NotRequired[str]  # type: ignore
