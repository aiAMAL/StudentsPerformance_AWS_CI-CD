from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    kaggle_dataset_id: str
    dataset_path: Path
    train_data_path: Path
    test_data_path: Path


@dataclass(frozen=True)
class FeaturesDataTransformation:
    target_variable: str
    numerical_features: List[str]
    categorical_features: List[str]

@dataclass(frozen=True)
class PipelineStepsTransformation:
    method_class: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class PipelineDataTransformation:
    numerical_pipeline: List[PipelineStepsTransformation] = field(default_factory=list)
    categorical_pipeline: List[PipelineStepsTransformation] = field(default_factory=list)


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    features_output_path: Path
    features_data_transformation: FeaturesDataTransformation
    pipeline_data_transformation: PipelineDataTransformation


@dataclass(frozen=True)
class RegressorConfig:
    model_class: str
    hyperparams: Dict[str, Union[float, int, str, List[Union[float, int, str]]]]


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_output_pkl: Path
    transformed_data_pkl: Path
    list_trained_models: List[RegressorConfig] = field(default_factory=dict)
