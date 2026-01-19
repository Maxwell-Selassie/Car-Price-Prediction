"""Module for cleaning and preprocessing the car details dataset.
Includes functions for handling missing values, removing duplicates,
and correcting data types.

Workflow:
1. Load necessary libraries.
2. Define functions for data type corrections.
3. Define functions for missing value handling.
4. Define functions for duplicate removal.
"""

# import necessary libraries
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from utils import LoggerMixin
import json

class DataCleaner(LoggerMixin):
    """Class for cleaning and preprocessing the car details dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.logger = self.setup_class_logger('DataCleaner',config,'logging')
        self.cleaning_results = {}

    def correct_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrects data types of specified columns in the DataFrame.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with corrected data types.
        """
        if df is None:
            self.logger.error("Input DataFrame is None.")
            raise ValueError("Input DataFrame cannot be None.")
        
        if 'dtype_corrections' not in self.config:
            self.logger.warning("No data type corrections specified in config.")
            return df
        
        convert_to_category = self.config['dtype_corrections'].get('convert_to_category', [])
        try:
            for column in convert_to_category:
                if column in df.columns:
                    df[column] = df[column].astype('category')
                    self.logger.info(f"Converted column {column} to 'category' dtype.")
                else:
                    self.logger.warning(f"Column {column} not found in DataFrame for dtype conversion.")

                self.cleaning_results['dtype_corrections'] = f"Converted columns to category: {convert_to_category}"
            return df
        
        except Exception as e:
            self.logger.error(f"Error during data type correction: {e}")
            raise e
        


    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles missing values in the DataFrame based on the specified strategy.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        if not 'preprocessing_steps' in self.config and 'missing_values' not in self.config['preprocessing_steps']:
            self.logger.warning("Missing value handling not specified in preprocessing steps.")
            raise ValueError("Missing value handling not specified in preprocessing steps.")
        
        if df is None:
            self.logger.error("Input DataFrame is None.")
            raise ValueError("Input DataFrame cannot be None.")
        
        if not self.config['preprocessing_steps']['missing_values']['enabled']:
            self.logger.warning("Missing value handling is disabled in config.")
            return df
        
        strategy = self.config['preprocessing_steps']['missing_values'].get('strategy', 'drop')
        try:
                if strategy == 'drop':
                    initial_shape = df.shape
                    df = df.dropna()
                    self.logger.info(f"Dropped missing values. Shape changed from {initial_shape} to {df.shape}.")
                    self.cleaning_results['missing_values'] = f"Dropped missing values. New shape: {df.shape}"
                
                else:
                    self.logger.error(f"Unknown missing value handling strategy: {strategy}")
                    raise ValueError(f"Unknown missing value handling strategy: {strategy}")
                
                return df
            
        except Exception as e:
                self.logger.error(f"Error during missing value handling: {e}")
                raise e
        

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicate rows from the DataFrame.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with duplicates removed.
        """
        if not 'preprocessing_steps' in self.config and 'duplicates' not in self.config['preprocessing_steps']:
            self.logger.warning("Duplicate removal not specified in preprocessing steps.")
            raise ValueError("Duplicate removal not specified in preprocessing steps.")
        
        if df is None:
            self.logger.error("Input DataFrame is None.")
            raise ValueError("Input DataFrame cannot be None.")
        
        if not self.config['preprocessing_steps']['duplicates']['enabled']:
            self.logger.warning("Duplicate removal is disabled in config.")
            return df
        
        strategy = self.config['preprocessing_steps']['duplicates'].get('strategy', 'drop')
        try:
            initial_shape = df.shape
            df = df.drop_duplicates()
            self.logger.info(f"Removed duplicates. Shape changed from {initial_shape} to {df.shape}.")
            self.cleaning_results['remove_duplicates'] = f"Removed duplicates. New shape: {df.shape}"
            return df
        
        except Exception as e:
            self.logger.error(f"Error during duplicate removal: {e}")
            raise e
        
    def save_cleaning_report(self, report_path: str) -> None:
        """Saves the cleaning report to a specified json file path.
        
        Args:
            report_path (str): The file path to save the cleaning report.
        """
        try:
            with open(report_path, 'w') as f:
                json.dump(self.cleaning_results, f, indent=4)
            self.logger.info(f"Cleaning report saved to {report_path}.")
        except Exception as e:
            self.logger.error(f"Error saving cleaning report: {e}")
            raise e
        
        
    def clean_data(self, df: pd.DataFrame, report_path: str) -> pd.DataFrame:
        """Cleans the data by applying all preprocessing steps in sequence.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            report_path (str): The file path to save the cleaning report.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        self.logger.info("Starting data cleaning process.")
        df = self.correct_data_types(df)
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        self.save_cleaning_report(report_path)
        self.logger.info("Data cleaning process completed.")
        return df   