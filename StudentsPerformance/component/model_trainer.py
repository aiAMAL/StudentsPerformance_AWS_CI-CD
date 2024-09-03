import sys
from pathlib import Path
from importlib import import_module
from sklearn.metrics import r2_score
from StudentsPerformance.utils import save_object
from StudentsPerformance.logger import logger
from StudentsPerformance.exception import CustomException
from StudentsPerformance.entity import ModelTrainerConfig

from StudentsPerformance.config import ConfigurationManager
from StudentsPerformance.component.data_ingestion import DataIngestion
from StudentsPerformance.component.data_transformation import DataTransformation

class ModelTraining:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initialize_model_class(self):
        list_of_models = self.config.list_trained_models
        classes = []
        for model in list_of_models:
            module_name, class_name = model.model_class.rsplit('.', 1)
            module = import_module(module_name)
            cls = getattr(module, class_name)

            # Initialize the class with the parameters
            class_instance = cls()
            # print(class_name, '  ', type(class_name))
            # print(class_instance, '  ', type(class_instance))
            classes.append({class_name: class_instance})
        return classes

    def evaluate_model(self, X_train, y_train, X_test, y_test, models):
        try:
            report_score = {}
            trained_models = {}
            for model in models:
                model_name = list(model.keys())[0]
                model_cls = list(model.values())[0]
                # print(f'\n\n--------- Train {model_name} -----------')
                # print()
                model_cls.fit(X_train, y_train)
                y_pred = model_cls.predict(X_test)
                test_model_score = r2_score(y_test, y_pred)
                # print('-->score: ', test_model_score)
                # report_score[model_name] = test_model_score
                # trained_models[model_name] = model_cls
                report_score[model_name] = (test_model_score, model_cls)
            return report_score # , trained_models
        except Exception as e:
            raise CustomException(e, sys)


    def modeling(self, train_array, test_array):
        X_train = train_array[:, : -1]
        y_train = train_array[:, -1]
        X_test = test_array[:, : -1]
        y_test = test_array[:, -1]

        instantiated_models = self.initialize_model_class()
        model_report = self.evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=instantiated_models)

        best_model_score, (best_model_name, best_model_cls) = max(model_report.items(), key=lambda item: item[1][0])    # apply max on first element[0] of the second variable(tupele)[1}

        print('---> best_model_score: ', best_model_score)
        print('---> best_model_name: ', best_model_name)
        print('---> best_model_cls: ', best_model_cls)
        save_object(file_path=Path(self.config.model_output_pkl), object=best_model_cls)



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

        # Model training process

        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTraining(model_trainer_config)
        model_trainer.modeling(train_arr, test_arr)

        # Model evaluation process
        #...

        # Model deployment process
        #...

        # Model monitoring process
        #...

        # Model maintenance process
        #...

        # Reporting process
        #...

        # Logging and error handling process
        #...

        # Final message
        print("Model training pipeline completed successfully.")

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Error in the main pipeline: {str(e)}")
        raise CustomException(e, sys)
