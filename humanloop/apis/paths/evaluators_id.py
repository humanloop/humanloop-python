from humanloop.paths.evaluators_id.get import ApiForget
from humanloop.paths.evaluators_id.delete import ApiFordelete
from humanloop.paths.evaluators_id.patch import ApiForpatch


class EvaluatorsId(
    ApiForget,
    ApiFordelete,
    ApiForpatch,
):
    pass
