import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from StudentsPerformance.entity import DataIngestionConfig
from StudentsPerformance.config import ConfigurationManager
from StudentsPerformance.exception import CustomException
from StudentsPerformance.logger import logger
from dotenv import load_dotenv
load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _initialize_kaggle_api(self) -> KaggleApi:
        """Initialize and authenticate Kaggle API."""
        api = KaggleApi()
        api.authenticate()
        return api

    def _download_dataset(self, api: KaggleApi):
        try:
            data_ingestion_path = Path(self.config.root_dir)
            data_identifier = self.config.kaggle_dataset_id

            # Download the dataset
            logger.info(f"Starting download for dataset: {data_identifier}")
            api.dataset_download_files(data_identifier, path=str(data_ingestion_path), unzip=True)
            logger.info(f"Dataset downloaded successfully to {data_ingestion_path}")

            # Rename the downloaded file
            downloaded_csv_file = Path(data_ingestion_path, os.listdir(data_ingestion_path)[0])
            downloaded_csv_file.rename(self.config.dataset_path)

        except Exception as e:
            logger.error(f"Error occurred while downloading dataset: {str(e)}")
            raise CustomException(e, sys)

    def _load_data(self) -> pd.DataFrame:
        """Load dataset from CSV file."""
        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.exists():
            logger.error(f'Dataset file not found: {dataset_path}')
            raise FileNotFoundError(f'Dataset file not found: {dataset_path}')
        return pd.read_csv(dataset_path)

    def _split_and_save_data(self, dataset: pd.DataFrame):
        try:
            # Splitting dataset into training and test sets
            train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

            train_data.to_csv(self.config.train_data_path, index=False, header=True)
            test_data.to_csv(self.config.test_data_path, index=False, header=True)
            logger.info('Saving training & testing dataset  -->  completed')

        except Exception as e:
            logger.error(f"Error occurred during data splitting and saving: {e}")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            api = self._initialize_kaggle_api()
            self._download_dataset(api)
            dataset = self._load_data()
            self._split_and_save_data(dataset)
            logger.info('Data ingestion process completed successfully')

        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise CustomException(e, sys)


if __name__ == '__main__':
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.initiate_data_ingestion()