# This file was auto-generated by Fern from our API Definition.

import typing
from ...requests.evaluator_version_id import EvaluatorVersionIdParams
from ...requests.evaluator_file_id import EvaluatorFileIdParams
from ...requests.evaluator_file_path import EvaluatorFilePathParams

AddEvaluatorsRequestEvaluatorsItemParams = typing.Union[
    EvaluatorVersionIdParams, EvaluatorFileIdParams, EvaluatorFilePathParams
]
