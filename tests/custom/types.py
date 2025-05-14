from typing import NamedTuple, Protocol

from humanloop import FileType
from humanloop.client import Humanloop


class GetHumanloopClientFn(Protocol):
    def __call__(self, use_local_files: bool = False) -> Humanloop: ...


class SyncableFile(NamedTuple):
    path: str
    type: FileType
    model: str
    id: str = ""
    version_id: str = ""
