import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from importlib import import_module
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from StudentsPerformance.logger import logger
from StudentsPerformance.utils import save_object
from StudentsPerformance.exception import CustomException
from StudentsPerformance.component.data_ingestion import DataIngestion
from StudentsPerformance.entity import DataTransformationConfig
from StudentsPerformance.config import ConfigurationManager


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initialize_pipeline(self):
        pipeline_data_transformation = self.config.pipeline_data_transformation

        def import_instantiate_pipeline_step(pipeline_step):
            """
            Imports and instantiates pipeline steps dynamically.

            :param pipeline_step: List of pipeline step configurations.
            :return: Initialized sklearn Pipeline object.
            """
            steps = []
            for step in pipeline_step:
                try:
                    # Dynamically import the class
                    module_name, class_name = step.method_class.rsplit('.', 1)
                    module = import_module(module_name)
                    cls = getattr(module, class_name)

                    # Initialize the class with the parameters
                    step_instance = cls(**step.params)
                    steps.append((class_name, step_instance))

                except ImportError as e:
                    logger.error(f"Module {module_name} could not be imported: {str(e)}")
                    raise CustomException(e, sys)
                except AttributeError as e:
                    logger.error(f"Class {class_name} not found in module {module_name}: {str(e)}")
                    raise CustomException(e, sys)
                except Exception as e:
                    logger.error(f"Error in {step.method_class} instantiation: {str(e)}")
                    raise CustomException(e, sys)

            return Pipeline(steps=steps)

        # Initialize the numerical pipeline
        yaml_numerical_pipeline = pipeline_data_transformation.numerical_pipeline
        numerical_pipeline = import_instantiate_pipeline_step(yaml_numerical_pipeline)

        # Initialize the categorical pipeline
        yaml_categorical_pipeline = pipeline_data_transformation.categorical_pipeline
        categorical_pipeline = import_instantiate_pipeline_step(yaml_categorical_pipeline)

        return numerical_pipeline, categorical_pipeline

    def data_transformation(self):
        try:
            # Load training and testing data
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)

            features = self.config.features_data_transformation
            numerical_features = features.numerical_features
            categorical_features = features.categorical_features

            # Separate features and target variable
            features_train = train_data.drop(columns=[features.target_variable], axis=1)
            features_test = test_data.drop(columns=[features.target_variable], axis=1)
            target_train = train_data[features.target_variable]
            target_test = test_data[features.target_variable]

            # Initialize the pipelines
            numerical_pipeline, categorical_pipeline = self.initialize_pipeline()

            # Combine the pipelines using ColumnTransformer
            processing = ColumnTransformer([('numerical_pipeline', numerical_pipeline, numerical_features),
                                            ('categorical_pipeline', categorical_pipeline, categorical_features)])

            # Transform the data
            train_features_transformed = processing.fit_transform(features_train)
            test_features_transformed = processing.transform(features_test)

            # Combine transformed features with target variable
            train_data_array = np.c_[train_features_transformed, np.array(target_train)]
            test_data_array = np.c_[test_features_transformed , np.array(target_test)]

            # Save the processing object for future use
            save_object(file_path=Path(self.config.features_output_path), object=processing)
            logger.info("Data transformation completed successfully.")

            return train_data_array, test_data_array

        except Exception as e:
            logger.error(f"Failed to perform data transformation: {str(e)}")
            raise CustomException(e, sys)


if __name__ == '__main__':
    try:
        # Configuration setup
        config = ConfigurationManager()

        # Data ingestion process
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.initiate_data_ingestion()

        # Data transformation process
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(data_transformation_config)
        train_arr, test_arr = data_transformation.data_transformation()

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Error in the main pipeline: {str(e)}")
        raise CustomException(e, sys)
