from humanloop.paths.projects_id.get import ApiForget
from humanloop.paths.projects_id.delete import ApiFordelete
from humanloop.paths.projects_id.patch import ApiForpatch


class ProjectsId(
    ApiForget,
    ApiFordelete,
    ApiForpatch,
):
    pass
