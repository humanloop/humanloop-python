from typing import Optional


class HumanloopRuntimeError(Exception):
    def __init__(self, message: Optional[str] = None):
        self.message = message

    def __str__(self) -> str:
        if self.message is None:
            return super().__str__()
        return self.message
