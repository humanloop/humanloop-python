import typing_extensions

from humanloop.apis.tags import TagValues
from humanloop.apis.tags.projects_api import ProjectsApi
from humanloop.apis.tags.evaluations_api import EvaluationsApi
from humanloop.apis.tags.datasets_api import DatasetsApi
from humanloop.apis.tags.logs_api import LogsApi
from humanloop.apis.tags.evaluators_api import EvaluatorsApi
from humanloop.apis.tags.model_configs_api import ModelConfigsApi
from humanloop.apis.tags.chats_api import ChatsApi
from humanloop.apis.tags.completions_api import CompletionsApi
from humanloop.apis.tags.datapoints_api import DatapointsApi
from humanloop.apis.tags.sessions_api import SessionsApi
from humanloop.apis.tags.feedback_api import FeedbackApi
from humanloop.apis.tags.authentication_api import AuthenticationApi

TagToApi = typing_extensions.TypedDict(
    'TagToApi',
    {
        TagValues.PROJECTS: ProjectsApi,
        TagValues.EVALUATIONS: EvaluationsApi,
        TagValues.DATASETS: DatasetsApi,
        TagValues.LOGS: LogsApi,
        TagValues.EVALUATORS: EvaluatorsApi,
        TagValues.MODEL_CONFIGS: ModelConfigsApi,
        TagValues.CHATS: ChatsApi,
        TagValues.COMPLETIONS: CompletionsApi,
        TagValues.DATAPOINTS: DatapointsApi,
        TagValues.SESSIONS: SessionsApi,
        TagValues.FEEDBACK: FeedbackApi,
        TagValues.AUTHENTICATION: AuthenticationApi,
    }
)

tag_to_api = TagToApi(
    {
        TagValues.PROJECTS: ProjectsApi,
        TagValues.EVALUATIONS: EvaluationsApi,
        TagValues.DATASETS: DatasetsApi,
        TagValues.LOGS: LogsApi,
        TagValues.EVALUATORS: EvaluatorsApi,
        TagValues.MODEL_CONFIGS: ModelConfigsApi,
        TagValues.CHATS: ChatsApi,
        TagValues.COMPLETIONS: CompletionsApi,
        TagValues.DATAPOINTS: DatapointsApi,
        TagValues.SESSIONS: SessionsApi,
        TagValues.FEEDBACK: FeedbackApi,
        TagValues.AUTHENTICATION: AuthenticationApi,
    }
)
