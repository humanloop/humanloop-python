from humanloop.paths.logs.get import ApiForget
from humanloop.paths.logs.post import ApiForpost
from humanloop.paths.logs.delete import ApiFordelete
from humanloop.paths.logs.patch import ApiForpatch


class Logs(
    ApiForget,
    ApiForpost,
    ApiFordelete,
    ApiForpatch,
):
    pass
