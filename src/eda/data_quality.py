""" Data quality functions for exploratory data analysis (EDA). 
Workflows:
1. Check for empty dataframe
2. Check for rows and columns
3. Check for negative values in numeric columns
4. Check for infinite values in numeric columns
5. Check for expected columns in the dataframe
6. Check for target column presence
7. Check for missing values in the dataframe
8. Check for duplicate rows in the dataframe
9. Check for data constants in the dataframe
10. Check for outliers in numeric columns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict
from utils import LoggerMixin, ensure_directory, write_json
import mlflow
import json 

class DataQualityChecker(LoggerMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = self.setup_class_logger("DataQualityChecker", config, "logging")
        self.output_dir = config["file_paths"].get("eda_reports", "data_quality_reports")
        ensure_directory(self.output_dir)
        self.validation: Dict[str, Any] = {}

    def empty_dataframe_check(self, df: pd.DataFrame) -> None:
        """Check if the dataframe is empty."""
        is_empty = df.empty
        self.validation['empty_dataframe'] = is_empty
        if is_empty:
            self.logger.warning("The dataframe is empty.")
            raise ValueError("The dataframe is empty.")
        else:
            self.logger.info("The dataframe is not empty.")

    def multiple_rows_columns_check(self, df: pd.DataFrame) -> None:
        """Check if the dataframe has multiple rows and columns."""
        num_rows, num_cols = df.shape
        has_multiple = num_rows > 1 and num_cols > 1
        self.validation['multiple_rows_columns'] = has_multiple
        if not has_multiple:
            self.logger.warning("The dataframe does not have multiple rows and columns.")
            raise ValueError("The dataframe must have multiple rows and columns.")
        else:
            self.logger.info("The dataframe has multiple rows and columns.")

    def negative_values_check(self, df: pd.DataFrame) -> None:
        """Check for negative values in numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        negative_values = {col: (df[col] < 0).any() for col in numeric_cols}
        self.validation['negative_values'] = negative_values
        for col, has_neg in negative_values.items():
            if has_neg:
                self.logger.warning(f"Column '{col}' contains negative values.")
        if any(negative_values.values()):
            raise ValueError("Some numeric columns contain negative values.")
        else:
            self.logger.info("No negative values found in numeric columns.")

    def infinite_values_check(self, df: pd.DataFrame) -> None:
        """Check for infinite values in numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        infinite_values = {col: np.isinf(df[col]).any() for col in numeric_cols}
        self.validation['infinite_values'] = infinite_values
        for col, has_inf in infinite_values.items():
            if has_inf:
                self.logger.warning(f"Column '{col}' contains infinite values.")
        if any(infinite_values.values()):
            raise ValueError("Some numeric columns contain infinite values.")
        else:
            self.logger.info("No infinite values found in numeric columns.")

    def expected_columns_check(self, df: pd.DataFrame, expected_columns: List[str]) -> None:
        """Check for expected columns in the dataframe."""
        missing_columns = [col for col in expected_columns if col not in df.columns]
        self.validation['missing_columns'] = missing_columns
        if missing_columns:
            self.logger.warning(f"Missing expected columns: {missing_columns}")
            raise ValueError(f"The dataframe is missing expected columns: {missing_columns}")
        else:
            self.logger.info("All expected columns are present in the dataframe.")

    def target_column_check(self, df: pd.DataFrame, target_column: str) -> None:
        """Check for the presence of the target column in the dataframe."""
        has_target = target_column in df.columns
        self.validation['target_column_present'] = has_target
        if not has_target:
            self.logger.warning(f"Target column '{target_column}' is missing.")
            raise ValueError(f"The target column '{target_column}' is missing from the dataframe.")
        else:
            self.logger.info(f"Target column '{target_column}' is present in the dataframe.")

    def check_data_constants(self, df: pd.DataFrame) -> List[str]:
        """
            Check for data constants in the dataframe.

            Args:
                df: pd.DataFrame

            Returns:
                List[str]: List of columns with constant values
        """
        constant_columns = [col for col in df.columns if df[col].nunique() == 1]
        if constant_columns:
            self.validation['constant_columns'] = constant_columns
            self.logger.warning(f'Found constant columns: {constant_columns}')
            raise ValueError(f'Constant columns found: {constant_columns}')
        else:
            self.logger.info('No constant columns found in the dataset.')

        self.validation['constant_columns'] = constant_columns

    # missing values check
    def missing_values_check(self, df: pd.DataFrame) -> None:
        """
            Check for missing values in the dataframe and return a summary dataframe
            containing columns with missing values, number of missing values and percentage
            of missing values.

            Args:
                df: pd.DataFrame
        """
        missing_summary = df.isnull().sum()
        if len( missing_summary[missing_summary > 0]) == 0:
            self.logger.info('No missing values found in the dataset.')
            return pd.DataFrame(columns=['Missing Values', 'Percentage'])
        
        # log parameters to mlflow
        missing_summary = missing_summary[missing_summary > 0]
        missing_percentage = (missing_summary / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Values': missing_summary,
            'Percentage': missing_percentage
        })
        artifact_path = f"{self.output_dir}/missing_values_summary.csv"
        missing_df.to_csv(artifact_path)

        mlflow.log_artifact(artifact_path)
        self.validation['total_missing_values'] = missing_summary.sum()

    #checking for duplicated data, keeping the first occurence
    def check_duplicates(self, df: pd.DataFrame) -> None:
        """
            Check for duplicated rows in the dataframe.

            Args:
                df: pd.DataFrame

        """
        n_duplicates = df.duplicated(keep='first').sum() #returns duplicated rows, keeping only the first occurence
        if n_duplicates > 0:
            self.logger.warning(f'Found {n_duplicates} duplicated rows in the dataset.')
        else:
            self.logger.info('No duplicated rows found in the dataset.')
        self.validation['n_duplicated_rows'] = n_duplicates

    def outliers_check(self, df: pd.DataFrame) -> None:
        """
            Check for outliers in numeric columns using the IQR method.

            Args:
                df: pd.DataFrame

        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_info[col] = {
                'num_outliers': len(outliers),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            artifact_path = f"{self.output_dir}/outliers.json"
            write_json(outlier_info, artifact_path, indent=4)
            if len(outliers) > 0:
                self.logger.warning(f"Column '{col}' contains {len(outliers)} outliers.")
        
        self.validation['outliers'] = outlier_info

    
    def run_all_checks(self, df: pd.DataFrame, expected_columns: List[str], 
                    target_column: str) -> Dict[str, Any]:
        """Run all data quality checks on the dataframe."""
        self.empty_dataframe_check(df)
        self.multiple_rows_columns_check(df)
        self.negative_values_check(df)
        self.infinite_values_check(df)
        self.expected_columns_check(df, expected_columns)
        self.target_column_check(df, target_column)
        self.missing_values_check(df)
        self.check_duplicates(df)
        self.check_data_constants(df)
        self.outliers_check(df)


    def save_validation_report(self) -> None:
        """Save the validation dict as json data to a specified path."""
        if self.validation:
            output_path = Path(f"{self.output_dir}/data_quality_validation_report.json")
            write_json(self.validation, output_path, indent=4)
            self.logger.info(f"Validation report saved to {output_path}")
        else:
            self.logger.warning("No validation data to save.")

        mlflow.log_artifact(str(output_path))
        