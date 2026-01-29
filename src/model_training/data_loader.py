"""  Loaders for various datasets for modelling
This function provides various functions to load training and dev sets

Workflows:
1. Import necessary libraries
2. Define dataset loading functions
"""

# import necessary libraries
import pandas as pd
import numpy as np
from pathlib import Path
from utils import LoggerMixin
from typing import Dict, List, Any, Tuple
import mlflow
import shutil
import json
class DataLoader(LoggerMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config.get('file_paths',{})
        self.logger = self.setup_class_logger("DataLoader_training",config, "logging")

        # data splits
        self.X_train = None
        self.y_train = None 

        self.feature_names = None
        self.n_features = None

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads train and dev sets
        
        Returns:
            Tuple(X_train, y_train)
        """
        self.logger.info(f'='*50)
        self.logger.info(f'LOADING DATA')
        self.logger.info(f'='*50)

            # load train set
        try:
            train_path = self.config.get('train_data','data/processed/train_processed_v1.csv')
            self.logger.info(f'Loading training set from: {train_path}')
            train_df = pd.read_csv(train_path)

            target_column = self.config.get('target_column','selling_price_log')
            if target_column not in train_df:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            
            self.X_train = train_df.drop(columns=[target_column])
            self.y_train = train_df[target_column]
            self.feature_names = train_df.drop(columns=[target_column]).columns.tolist()
            self.n_features = len(self.feature_names)

            self.logger.info(f"Train: X={self.X_train.shape}, y={self.y_train.shape}")
            self.logger.info(f"Features: {self.n_features}")

            
        except Exception as e:
            self.logger.error(f"Error loading train sets: {e}")
            raise e
            

    def get_metadata_and_save_metadata(self) -> Dict[str,Any]:
            """ Get dataset metadata

            Returns:
                Dictionary with dataset metadata
            """
            metadata = {
                'n_features' : self.n_features,
                'feature_names' : self.feature_names,
                'train_size' : len(self.X_train)
            }

            output_dir = self.config.get('metrics_artifacts','artifacts/metrics')
            metrics_path = Path(output_dir) / f'data_metadata.json'

            if metrics_path.is_dir():
                shutil.rmtree(metrics_path)

            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(metrics_path, 'w') as file:
                    json.dump(metadata, file, indent=4)
                    self.logger.info(f"Data metadata saved to: {metrics_path}")

            except Exception as e:
                self.logger.error(f"Error saving data to json file: {e}")
                raise e
            mlflow.log_artifact(metrics_path)