"""  
Loaders for various datasets used in exploratory data analysis (EDA).
This module provides functions to load and preprocess datasets for analysis.

workflow:
1. Define dataset loading functions.    
"""

import pandas as pd
import numpy  as np
from pathlib import Path
import mlflow
from typing import Any, Dict, List
from utils import LoggerMixin, read_csv

class DataLoader(LoggerMixin):
    """Class for loading datasets used in EDA."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = self.setup_class_logger('DataLoader', config, 'logging')

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a dataset by name.

        Args:
            dataset_name (str): Name of the dataset to load.

        Returns:
            pd.DataFrame: Loaded dataset as a pandas DataFrame.
        """
        try:
            self.logger.info(f"Loading dataset: {dataset_name}")
            dataset_path = Path(self.config['file_paths']['raw_data'])
            df = read_csv(dataset_path)

            dataset = mlflow.data.from_pandas(
                df, source=dataset_path
            )


            mlflow.log_input(dataset, context="primary_dataset")
            mlflow.set_tag("mlflow.note.content",
            f"Loaded {dataset_name} dataset from {dataset_path} for EDA."
            f"Data contains {df.shape[0]} rows and {df.shape[1]} columns.")

            self.logger.info(f"Dataset {dataset_name} loaded with shape {df.shape}")

            return df
        except Exception as e:
            self.logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise

