from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    kaggle_dataset_id: str
    dataset_path: Path
    train_data_path: Path
    test_data_path: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    features_output_path: Path
    categorical_features: List[str]
    numerical_features: List[str]
    target_variable: str

