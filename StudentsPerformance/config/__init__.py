from typing import Dict, List, Any
from pathlib import Path
from StudentsPerformance.utils import read_yaml, create_directories
from StudentsPerformance.entity import (DataIngestionConfig,
                                        DataTransformationConfig,
                                        PipelineStepsTransformation,
                                        PipelineDataTransformation,
                                        FeaturesDataTransformation)


CONFIG_FILE_PATH = Path('config/config.yaml')


class ConfigurationManager:
    def __init__(self, config_filepath: Path = CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)
        create_directories([Path(self.config.artifact_root)])

    # ====================================================================
    # -------------------------- Data Ingestion --------------------------
    # ====================================================================

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([Path(config.root_dir)])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            kaggle_dataset_id=config.kaggle_dataset_id,
            dataset_path=config.dataset_path,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path
        )

        return data_ingestion_config

    # ====================================================================
    # ----------------------- Data Transformation ------------------------
    # ====================================================================

    def get_features_data_transformation(self) -> FeaturesDataTransformation:
        config = self.config.features_data_transformation
        features_data_transformation = FeaturesDataTransformation(
            target_variable = config.target_variable,
            numerical_features = config.numerical_features,
            categorical_features = config.categorical_features
        )

        return features_data_transformation

    def get_pipeline_steps_transformation(self, pipeline_config: List[Dict[str, Dict[str, Any]]]) -> List[PipelineStepsTransformation]:
        steps = []
        for step in pipeline_config:
            for method_class, params in step.items():
                steps.append(
                    PipelineStepsTransformation(method_class=method_class, params=params)
                )

        return steps

    def get_pipeline_data_transformation(self) -> PipelineDataTransformation:
        config = self.config.pipeline_data_transformation

        return PipelineDataTransformation(
            numerical_pipeline=self.get_pipeline_steps_transformation(config.numerical_pipeline),
            categorical_pipeline=self.get_pipeline_steps_transformation(config.categorical_pipeline)
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([Path(config.root_dir)])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            features_output_path=config.features_output_path,
            features_data_transformation=self.get_features_data_transformation(),
            pipeline_data_transformation=self.get_pipeline_data_transformation()
        )

        return data_transformation_config
