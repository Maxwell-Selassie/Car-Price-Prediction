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


class DataLoader(LoggerMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config.get('file_paths',{})
        self.logger = self.setup_class_logger("DataLoader_training",config, "logging")

        # data splits
        self.X_train = None
        self.y_train = None 
        self.X_dev = None 
        self.y_dev = None 

        self.feature_names = None
        self.n_features = None

    def load(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """Loads train and dev sets
        
        Returns:
            Tuple(X_train, y_train, X_dev, y_dev)
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
            
            self.X_train = train_df.drop(columns=[target_column]).to_numpy()
            self.y_train = train_df[target_column].to_numpy()
            self.feature_names = train_df.drop(columns=[target_column]).columns.tolist()
            self.n_features = len(self.feature_names)

            self.logger.info(f"Train: X={self.X_train.shape}, y={self.y_train.shape}")
            self.logger.info(f"Features: {self.n_features}")


            # Load validation set
            dev_path = self.config.get('dev_data',{})
            self.logger.info(f'Loading development set from: {dev_path}')
            dev_df = pd.read_csv(dev_path)

            self.X_dev = dev_df.drop(columns=[dev_path]).to_numpy()
            self.y_dev = dev_df[target_column].to_numpy()

            self.logger.info(f'Dev: X={self.X_dev.shape}, y={self.y_dev.shape}')

            # validate shapes
            if self.X_train.shape[1] != self.X_dev.shape[1]:
                raise ValueError(f'Feature miscount between train and dev sets')
            
        except Exception as e:
            self.logger.error(f"Error loading train/dev sets: {e}")
            raise e
            

    def get_metadata(self) -> Dict[str,Any]:
            """ Get dataset metadata

            Returns:
                Dictionary with dataset metadata
            """
            return {
                'n_features' : self.n_features,
                'feature_names' : self.feature_names,
                'train_size' : len(self.X_train),
                'dev_size' : len(self.X_dev)
            }