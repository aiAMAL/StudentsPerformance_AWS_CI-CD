import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from StudentsPerformance.utils import save_object
from StudentsPerformance.component.data_ingestion import DataIngestion
from StudentsPerformance.entity import DataTransformationConfig
from StudentsPerformance.config import ConfigurationManager
from StudentsPerformance.exception import CustomException
from StudentsPerformance.logger import logger

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def data_transformation(self):
        numerical_features = self.config.numerical_features
        categorical_features = self.config.categorical_features
        target_variable = self.config.target_variable

        numerical_pipeline = Pipeline(
            steps=[('SimpleImputer', SimpleImputer(strategy='median')),
                   ('StandardScaler', StandardScaler(with_mean=False))]
        )

        categorical_pipeline = Pipeline(
            steps=[('SimpleImputer', SimpleImputer(strategy='most_frequent')),
                   ('OneHotEncoder', OneHotEncoder(sparse_output=False)),
                   ('StandardScaler', StandardScaler(with_mean=False))]
        )

        processing = ColumnTransformer([('numerical_pipeline', numerical_pipeline, numerical_features),
                                        ('categorical_pipeline', categorical_pipeline, categorical_features)])

        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        features_train_data = train_data.drop(columns=[self.config.target_variable], axis=1)
        target_train_data = train_data[self.config.target_variable]

        features_test_data = test_data.drop(columns=[self.config.target_variable], axis=1)
        target_test_data = test_data[self.config.target_variable]

        features_train_data_array = processing.fit_transform(features_train_data)
        features_test_data_array = processing.transform(features_test_data)

        train_array = np.c_[features_train_data_array, np.array(target_train_data)]
        test_array = np.c_[features_test_data_array, np.array(target_test_data)]

        save_object(
            file_path=Path(self.config.features_output_path),
            object=processing
        )

        return (
            train_array,
            test_array
        )


if __name__ == '__main__':
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.initiate_data_ingestion()
    data_transformation_config = config.get_data_transformation_config()
    data_transformation = DataTransformation(data_transformation_config)
    train_arr, test_arr = data_transformation.data_transformation()

