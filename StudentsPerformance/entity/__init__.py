from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    kaggle_dataset_id: str
    dataset_path: Path
    train_data_path: Path
    test_data_path: Path

