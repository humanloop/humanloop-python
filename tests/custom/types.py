from typing import Protocol, NamedTuple
from humanloop.client import Humanloop
from humanloop import FileType


class GetHumanloopClientFn(Protocol):
    def __call__(self, use_local_files: bool = False) -> Humanloop: ...


class SyncableFile(NamedTuple):
    path: str
    type: FileType
    model: str
    id: str = ""
    version_id: str = ""
