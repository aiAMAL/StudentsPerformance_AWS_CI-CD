from pathlib import Path
from StudentsPerformance.utils import read_yaml, create_directories
from StudentsPerformance.entity import (DataIngestionConfig,
                                        DataTransformationConfig)


CONFIG_FILE_PATH = Path('config/config.yaml')


class ConfigurationManager:
    def __init__(self, config_filepath: Path = CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)
        create_directories([Path(self.config.artifact_root)])

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

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        # create_directories([Path(config.root_dir)])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            features_output_path=config.features_output_path,
            categorical_features=config.categorical_features,
            numerical_features=config.numerical_features,
            target_variable=config.target_variable
        )

        return data_transformation_config
