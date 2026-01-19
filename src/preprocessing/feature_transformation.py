"""Feature transformation utilities for preprocessing.

Workflow:
1. Import necessary libraries.
2. Define function for log transformation and drop original column if specified.
3. Define function for square root transformation and drop original column if specified.
4. Define main transformation function to apply selected transformations.
5. Save transformation results.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any
from utils import LoggerMixin

class FeatureTransformer(LoggerMixin):
    """Class for transforming features in the dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.logger = self.setup_class_logger('FeatureTransformer', config, 'logging')
        self.transformation_result = {}

    def log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies log transformation to specified columns.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with log-transformed columns if there are columns to log transform, else None.
        """
        if df is None:
            self.logger.error("Input DataFrame is None.")
            raise ValueError("Input DataFrame cannot be None.")
        
        if not 'preprocessing_steps' in self.config and 'transformation' not in self.config['preprocessing_steps'] and 'log_transform' not in self.config['preprocessing_steps']['transformation']:
            self.logger.info("Log transformation not specified in config. Skipping.")
            return df
        
        columns_for_log_transform = self.config['preprocessing_steps']['transformation']['log_transform'].get('columns', [])
        try:
            for column in columns_for_log_transform:
                if column in df.columns:
                    df[f'{column}_log'] = np.log1p(df[column])
                    if self.config['preprocessing_steps']['transformation']['log_transform'].get('drop_original', True):
                        df.drop(columns=[column], inplace=True)
                        self.logger.info(f"Dropped original column {column} after log transformation.")
                    self.logger.info(f"Applied log transformation to column {column}.")
                else:
                    self.logger.warning(f"Column {column} not found in DataFrame for log transformation.")
            self.transformation_result['log_transform'] = 'Success'
            return df

        except Exception as e:
            self.logger.error(f'Error log-transforming columns: {e}')
            self.transformation_result['log_transform'] = 'Fail'
            raise e 
        
    def square_root_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies square transformation to specified columns.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with squared columns if there are columns to square, else None.
        """
        if df is None:
            self.logger.error("Input DataFrame is None")
            raise ValueError("Input DataFrame cannot be None")
        
        if not 'preprocessing_steps' in self.config and 'transformation' not in self.config['preprocessing_steps'] and 'sqrt_transform' not in self.config['preprocessing_steps']['transformation']:
            self.logger.info(f'Square transformations not specified in config. Skipping')
            return df
        
        columns_for_square = self.config['preprocessing_steps']['transformation']['sqrt_transform'].get('columns', [])
        try:
            for column in columns_for_square:
                if column in df.columns:
                    df[f'{column}_sqrt'] = np.sqrt(df[column])
                    if self.config['preprocessing_steps']['transformation']['sqrt_transform'].get('drop_original', True):
                        df.drop(columns=[column], inplace=True)
                        self.logger.info(f"Dropped original column {column} after square root transformation.")
                    self.logger.info(f"Applied square root transformation to column {column}.")
                else:
                    self.logger.warning(f"Column {column} not found in DataFrame for square root transformation.")

            self.transformation_result['sqrt_transformation'] = 'Success'
            return df
        
        except Exception as e:
            self.logger.error(f"Error sqrt-transforming columns: {e}")
            self.transformation_result['sqrt_tranformation'] = 'Fail'
            raise e

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main function to apply selected transformations based on config.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
        
        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        if df is None:
            self.logger.error("Input DataFrame is None.")
            raise ValueError("Input DataFrame cannot be None.")
        
        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input is not a valid DataFrame.")
            raise TypeError("Input must be a pandas DataFrame.")
        
        try:
            self.logger.info("Starting feature transformation process.")
            df = self.log_transform(df)
            df = self.square_root_transform(df)
            self.logger.info("Feature transformation process completed.")
            return df
        
        except Exception as e:
            self.logger.error(f"Error during feature transformation: {e}")
            raise e

    def save_transformation_report(self, report_path: str) -> None:
        """Saves the transformation results to a specified path.
        
        Args:
            report_path (str): The file path to save the transformation report.
        """
        try:
            with open(report_path, 'w') as f:
                json.dump(self.transformation_result, f, indent=4)
            self.logger.info(f"Transformation report saved to {report_path}.")
        except Exception as e:
            self.logger.error(f"Error saving transformation report: {e}")
            raise e