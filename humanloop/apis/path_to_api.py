import typing_extensions

from humanloop.paths import PathValues
from humanloop.apis.paths.completion import Completion
from humanloop.apis.paths.completion_deployed import CompletionDeployed
from humanloop.apis.paths.completion_experiment import CompletionExperiment
from humanloop.apis.paths.completion_model_config import CompletionModelConfig
from humanloop.apis.paths.chat import Chat
from humanloop.apis.paths.chat_deployed import ChatDeployed
from humanloop.apis.paths.chat_experiment import ChatExperiment
from humanloop.apis.paths.chat_model_config import ChatModelConfig
from humanloop.apis.paths.logs import Logs
from humanloop.apis.paths.logs_id import LogsId
from humanloop.apis.paths.feedback import Feedback
from humanloop.apis.paths.projects import Projects
from humanloop.apis.paths.projects_id import ProjectsId
from humanloop.apis.paths.projects_id_configs import ProjectsIdConfigs
from humanloop.apis.paths.projects_id_active_config import ProjectsIdActiveConfig
from humanloop.apis.paths.projects_id_active_experiment import ProjectsIdActiveExperiment
from humanloop.apis.paths.projects_id_feedback_types import ProjectsIdFeedbackTypes
from humanloop.apis.paths.projects_id_export import ProjectsIdExport
from humanloop.apis.paths.projects_id_deployed_configs import ProjectsIdDeployedConfigs
from humanloop.apis.paths.projects_project_id_deploy_config import ProjectsProjectIdDeployConfig
from humanloop.apis.paths.projects_project_id_deployed_config_environment_id import ProjectsProjectIdDeployedConfigEnvironmentId
from humanloop.apis.paths.model_configs import ModelConfigs
from humanloop.apis.paths.model_configs_id import ModelConfigsId
from humanloop.apis.paths.model_configs_id_export import ModelConfigsIdExport
from humanloop.apis.paths.model_configs_serialize import ModelConfigsSerialize
from humanloop.apis.paths.model_configs_deserialize import ModelConfigsDeserialize
from humanloop.apis.paths.projects_project_id_experiments import ProjectsProjectIdExperiments
from humanloop.apis.paths.experiments_experiment_id import ExperimentsExperimentId
from humanloop.apis.paths.experiments_experiment_id_model_config import ExperimentsExperimentIdModelConfig
from humanloop.apis.paths.sessions import Sessions
from humanloop.apis.paths.sessions_id import SessionsId
from humanloop.apis.paths.evaluators import Evaluators
from humanloop.apis.paths.evaluators_id import EvaluatorsId
from humanloop.apis.paths.datapoints_id import DatapointsId
from humanloop.apis.paths.datapoints import Datapoints
from humanloop.apis.paths.projects_project_id_datasets import ProjectsProjectIdDatasets
from humanloop.apis.paths.datasets_id import DatasetsId
from humanloop.apis.paths.datasets_dataset_id_datapoints import DatasetsDatasetIdDatapoints
from humanloop.apis.paths.evaluations_id import EvaluationsId
from humanloop.apis.paths.evaluations_id_datapoints import EvaluationsIdDatapoints
from humanloop.apis.paths.projects_project_id_evaluations import ProjectsProjectIdEvaluations
from humanloop.apis.paths.evaluations_evaluation_id_log import EvaluationsEvaluationIdLog
from humanloop.apis.paths.evaluations_evaluation_id_result import EvaluationsEvaluationIdResult
from humanloop.apis.paths.evaluations_id_status import EvaluationsIdStatus
from humanloop.apis.paths.evaluations_id_evaluators import EvaluationsIdEvaluators
from humanloop.apis.paths.evaluations import Evaluations
from humanloop.apis.paths.projects_project_id_finetunes import ProjectsProjectIdFinetunes
from humanloop.apis.paths.projects_project_id_finetunes_summary import ProjectsProjectIdFinetunesSummary
from humanloop.apis.paths.finetunes_id import FinetunesId

PathToApi = typing_extensions.TypedDict(
    'PathToApi',
    {
        PathValues.COMPLETION: Completion,
        PathValues.COMPLETIONDEPLOYED: CompletionDeployed,
        PathValues.COMPLETIONEXPERIMENT: CompletionExperiment,
        PathValues.COMPLETIONMODELCONFIG: CompletionModelConfig,
        PathValues.CHAT: Chat,
        PathValues.CHATDEPLOYED: ChatDeployed,
        PathValues.CHATEXPERIMENT: ChatExperiment,
        PathValues.CHATMODELCONFIG: ChatModelConfig,
        PathValues.LOGS: Logs,
        PathValues.LOGS_ID: LogsId,
        PathValues.FEEDBACK: Feedback,
        PathValues.PROJECTS: Projects,
        PathValues.PROJECTS_ID: ProjectsId,
        PathValues.PROJECTS_ID_CONFIGS: ProjectsIdConfigs,
        PathValues.PROJECTS_ID_ACTIVECONFIG: ProjectsIdActiveConfig,
        PathValues.PROJECTS_ID_ACTIVEEXPERIMENT: ProjectsIdActiveExperiment,
        PathValues.PROJECTS_ID_FEEDBACKTYPES: ProjectsIdFeedbackTypes,
        PathValues.PROJECTS_ID_EXPORT: ProjectsIdExport,
        PathValues.PROJECTS_ID_DEPLOYEDCONFIGS: ProjectsIdDeployedConfigs,
        PathValues.PROJECTS_PROJECT_ID_DEPLOYCONFIG: ProjectsProjectIdDeployConfig,
        PathValues.PROJECTS_PROJECT_ID_DEPLOYEDCONFIG_ENVIRONMENT_ID: ProjectsProjectIdDeployedConfigEnvironmentId,
        PathValues.MODELCONFIGS: ModelConfigs,
        PathValues.MODELCONFIGS_ID: ModelConfigsId,
        PathValues.MODELCONFIGS_ID_EXPORT: ModelConfigsIdExport,
        PathValues.MODELCONFIGS_SERIALIZE: ModelConfigsSerialize,
        PathValues.MODELCONFIGS_DESERIALIZE: ModelConfigsDeserialize,
        PathValues.PROJECTS_PROJECT_ID_EXPERIMENTS: ProjectsProjectIdExperiments,
        PathValues.EXPERIMENTS_EXPERIMENT_ID: ExperimentsExperimentId,
        PathValues.EXPERIMENTS_EXPERIMENT_ID_MODELCONFIG: ExperimentsExperimentIdModelConfig,
        PathValues.SESSIONS: Sessions,
        PathValues.SESSIONS_ID: SessionsId,
        PathValues.EVALUATORS: Evaluators,
        PathValues.EVALUATORS_ID: EvaluatorsId,
        PathValues.DATAPOINTS_ID: DatapointsId,
        PathValues.DATAPOINTS: Datapoints,
        PathValues.PROJECTS_PROJECT_ID_DATASETS: ProjectsProjectIdDatasets,
        PathValues.DATASETS_ID: DatasetsId,
        PathValues.DATASETS_DATASET_ID_DATAPOINTS: DatasetsDatasetIdDatapoints,
        PathValues.EVALUATIONS_ID: EvaluationsId,
        PathValues.EVALUATIONS_ID_DATAPOINTS: EvaluationsIdDatapoints,
        PathValues.PROJECTS_PROJECT_ID_EVALUATIONS: ProjectsProjectIdEvaluations,
        PathValues.EVALUATIONS_EVALUATION_ID_LOG: EvaluationsEvaluationIdLog,
        PathValues.EVALUATIONS_EVALUATION_ID_RESULT: EvaluationsEvaluationIdResult,
        PathValues.EVALUATIONS_ID_STATUS: EvaluationsIdStatus,
        PathValues.EVALUATIONS_ID_EVALUATORS: EvaluationsIdEvaluators,
        PathValues.EVALUATIONS: Evaluations,
        PathValues.PROJECTS_PROJECT_ID_FINETUNES: ProjectsProjectIdFinetunes,
        PathValues.PROJECTS_PROJECT_ID_FINETUNES_SUMMARY: ProjectsProjectIdFinetunesSummary,
        PathValues.FINETUNES_ID: FinetunesId,
    }
)

path_to_api = PathToApi(
    {
        PathValues.COMPLETION: Completion,
        PathValues.COMPLETIONDEPLOYED: CompletionDeployed,
        PathValues.COMPLETIONEXPERIMENT: CompletionExperiment,
        PathValues.COMPLETIONMODELCONFIG: CompletionModelConfig,
        PathValues.CHAT: Chat,
        PathValues.CHATDEPLOYED: ChatDeployed,
        PathValues.CHATEXPERIMENT: ChatExperiment,
        PathValues.CHATMODELCONFIG: ChatModelConfig,
        PathValues.LOGS: Logs,
        PathValues.LOGS_ID: LogsId,
        PathValues.FEEDBACK: Feedback,
        PathValues.PROJECTS: Projects,
        PathValues.PROJECTS_ID: ProjectsId,
        PathValues.PROJECTS_ID_CONFIGS: ProjectsIdConfigs,
        PathValues.PROJECTS_ID_ACTIVECONFIG: ProjectsIdActiveConfig,
        PathValues.PROJECTS_ID_ACTIVEEXPERIMENT: ProjectsIdActiveExperiment,
        PathValues.PROJECTS_ID_FEEDBACKTYPES: ProjectsIdFeedbackTypes,
        PathValues.PROJECTS_ID_EXPORT: ProjectsIdExport,
        PathValues.PROJECTS_ID_DEPLOYEDCONFIGS: ProjectsIdDeployedConfigs,
        PathValues.PROJECTS_PROJECT_ID_DEPLOYCONFIG: ProjectsProjectIdDeployConfig,
        PathValues.PROJECTS_PROJECT_ID_DEPLOYEDCONFIG_ENVIRONMENT_ID: ProjectsProjectIdDeployedConfigEnvironmentId,
        PathValues.MODELCONFIGS: ModelConfigs,
        PathValues.MODELCONFIGS_ID: ModelConfigsId,
        PathValues.MODELCONFIGS_ID_EXPORT: ModelConfigsIdExport,
        PathValues.MODELCONFIGS_SERIALIZE: ModelConfigsSerialize,
        PathValues.MODELCONFIGS_DESERIALIZE: ModelConfigsDeserialize,
        PathValues.PROJECTS_PROJECT_ID_EXPERIMENTS: ProjectsProjectIdExperiments,
        PathValues.EXPERIMENTS_EXPERIMENT_ID: ExperimentsExperimentId,
        PathValues.EXPERIMENTS_EXPERIMENT_ID_MODELCONFIG: ExperimentsExperimentIdModelConfig,
        PathValues.SESSIONS: Sessions,
        PathValues.SESSIONS_ID: SessionsId,
        PathValues.EVALUATORS: Evaluators,
        PathValues.EVALUATORS_ID: EvaluatorsId,
        PathValues.DATAPOINTS_ID: DatapointsId,
        PathValues.DATAPOINTS: Datapoints,
        PathValues.PROJECTS_PROJECT_ID_DATASETS: ProjectsProjectIdDatasets,
        PathValues.DATASETS_ID: DatasetsId,
        PathValues.DATASETS_DATASET_ID_DATAPOINTS: DatasetsDatasetIdDatapoints,
        PathValues.EVALUATIONS_ID: EvaluationsId,
        PathValues.EVALUATIONS_ID_DATAPOINTS: EvaluationsIdDatapoints,
        PathValues.PROJECTS_PROJECT_ID_EVALUATIONS: ProjectsProjectIdEvaluations,
        PathValues.EVALUATIONS_EVALUATION_ID_LOG: EvaluationsEvaluationIdLog,
        PathValues.EVALUATIONS_EVALUATION_ID_RESULT: EvaluationsEvaluationIdResult,
        PathValues.EVALUATIONS_ID_STATUS: EvaluationsIdStatus,
        PathValues.EVALUATIONS_ID_EVALUATORS: EvaluationsIdEvaluators,
        PathValues.EVALUATIONS: Evaluations,
        PathValues.PROJECTS_PROJECT_ID_FINETUNES: ProjectsProjectIdFinetunes,
        PathValues.PROJECTS_PROJECT_ID_FINETUNES_SUMMARY: ProjectsProjectIdFinetunesSummary,
        PathValues.FINETUNES_ID: FinetunesId,
    }
)
