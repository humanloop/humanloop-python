from typing import Optional


class HumanloopRuntimeError(Exception):
    """
    SDK custom code handles exceptions by populating Logs' `error` field.

    This exception signals an error severe enough to crash the execution
    e.g. illegal use of decorators.
    """

    def __init__(self, message: Optional[str] = None):
        self.message = message

    def __str__(self) -> str:
        if self.message is None:
            return super().__str__()
        return self.message
