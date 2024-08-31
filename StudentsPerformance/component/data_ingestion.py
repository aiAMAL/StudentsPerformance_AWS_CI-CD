import os
import sys
from pathlib import Path

import pandas as pd
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

    def load_data(self):
        try:
            api = KaggleApi()
            api.authenticate()

            data_ingestion_path = self.config.root_dir
            data_identifier = self.config.kaggle_dataset_id
            api.dataset_download_files(data_identifier, path=str(data_ingestion_path), unzip=True)

            csv_file = Path(data_ingestion_path, os.listdir(data_ingestion_path)[0])
            df = pd.read_csv(csv_file)

            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            train_data.to_csv(self.config.train_data_path, index=False, header=True)
            test_data.to_csv(self.config.test_data_path, index=False, header=True)
        except Exception as e:
            CustomException(e, sys)







if __name__ == '__main__':
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.load_data()