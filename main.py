import sys
from StudentsPerformance.logger import logger
from StudentsPerformance.exception import CustomException
from StudentsPerformance.config import ConfigurationManager
from StudentsPerformance.component.data_ingestion import DataIngestion
from StudentsPerformance.component.data_transformation import DataTransformation


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