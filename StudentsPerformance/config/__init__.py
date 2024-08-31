from pathlib import Path
from StudentsPerformance.utils import read_yaml, create_directories
from StudentsPerformance.entity import DataIngestionConfig

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

