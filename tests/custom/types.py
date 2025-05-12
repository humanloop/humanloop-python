from typing import Protocol
from humanloop.client import Humanloop

class GetHumanloopClientFn(Protocol):
    def __call__(self, use_local_files: bool = False) -> Humanloop: ...
