import inspect
from typing import Any, Callable


def args_to_inputs(func: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Maps arguments to their corresponding parameter names in the function signature.

    For example:
    ```python
    def foo(a, b=2, c=3):
        pass

    assert args_to_inputs(foo, (1, 2), {}) == {'a': 1, 'b': 2, 'c': 3}
    assert args_to_inputs(foo, (1,), {'b': 8}) == {'a': 1, 'b': 8, 'c': 3}
    assert args_to_inputs(foo, (1,), {}) == {'a': 1, 'b': 2, 'c': 3}
    ```
    """
    signature = inspect.signature(func)
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return dict(bound_args.arguments)
