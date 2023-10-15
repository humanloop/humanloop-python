from humanloop.paths.datasets_id.get import ApiForget
from humanloop.paths.datasets_id.delete import ApiFordelete
from humanloop.paths.datasets_id.patch import ApiForpatch


class DatasetsId(
    ApiForget,
    ApiFordelete,
    ApiForpatch,
):
    pass
