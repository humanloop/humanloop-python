import inspect
from typing import Any


def args_to_inputs(func: callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    signature = inspect.signature(func)
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return dict(bound_args.arguments)
