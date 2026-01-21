"""Module for cleaning and preprocessing the car details dataset.
Includes functions for handling missing values, removing duplicates,
and correcting data types.

Workflow:
1. Load necessary libraries.
2. Define functions for data type corrections.
3. Define functions for missing value handling.
4. Define functions for duplicate removal.
5. Extract age from year column.
6. Drop columns not needed for modeling.
7. Save a report of cleaning steps taken.
8. Define a main cleaning function to apply all steps in sequence.
"""

# import necessary libraries
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from utils import LoggerMixin, ensure_directory
import json
import mlflow

class DataCleaner(LoggerMixin):
    """Class for cleaning and preprocessing the car details dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
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
        if not 'preprocessing_steps' in self.config or 'missing_values' not in self.config['preprocessing_steps']:
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
        if not 'preprocessing_steps' in self.config or 'duplicates' not in self.config['preprocessing_steps']:
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
        
    def extract_age_from_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts age from the year column and adds it as a new column.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with age column added.
        """
        if 'year' not in df.columns:
            self.logger.error("Column 'year' not found in DataFrame.")
            raise ValueError("Column 'year' must be present in the DataFrame to extract age.")
        try:
            max_year_in_data = df['year'].max()
            df['age'] = max_year_in_data - df['year']
            self.logger.info("Extracted age from year and added as new column 'age'.")
            self.cleaning_results['extract_age'] = "Extracted age from year column."
            return df
        except Exception as e:
            self.logger.error(f"Error extracting age from year: {e}")
            raise e
        
    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops specified columns from the DataFrame.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with specified columns dropped.
        """ 
        if 'drop_columns' not in self.config:
            self.logger.warning("No columns specified to drop in config.")
            return df
        
        columns_to_drop = self.config['drop_columns']
        try:
            df = df.drop(columns=columns_to_drop, errors='ignore')
            self.logger.info(f"Dropped columns: {columns_to_drop}.")
            self.cleaning_results['drop_columns'] = f"Dropped columns: {columns_to_drop}"
            return df
        
        except Exception as e:
            self.logger.error(f"Error dropping columns: {e}")
            raise e

        
    def save_cleaning_report(self, name: str) -> None:
        """Saves the cleaning report to a specified json file path.
        """
        try:
            output_dir = self.config['save_artifacts'].get('report_path', 'artifacts/reports/')
            report_path = Path(output_dir) / f'{name}_cleaning_report.json'

            if report_path.is_dir():
                shutil.rmtree(report_path)

            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(self.cleaning_results, f, indent=4)

            mlflow.log_artifact(report_path)
            self.logger.info(f"Cleaning report saved to {report_path}.")
        except Exception as e:
            self.logger.error(f"Error saving cleaning report: {e}")
            raise e
        
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the data by applying all preprocessing steps in sequence.
        
        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        self.logger.info("Starting data cleaning process.")
        df = self.correct_data_types(df)
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        df = self.extract_age_from_year(df)
        df = self.drop_columns(df)
        self.cleaning_results['final_shape_after_cleaning'] = df.shape
        self.logger.info("Data cleaning process completed.")
        return df   