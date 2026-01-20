"""  
Module for splitting data into training, development and testing sets.

Workflow:
1. Import necessary libraries.
2. Define a function to split the data
into train, dev and test sets.
3. Return the split datasets for further preprocessing
Test size = 1000 samples
Dev size = 1000 samples
Train size = Remaining samples
4. Save the split datasets as CSV files for future use.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
from utils import LoggerMixin
from pathlib import Path
import mlflow
import shutil

class DataSplitter(LoggerMixin):
    """Class for splitting data into training, development and testing sets."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = self.setup_class_logger('DataSplitter',config,'logging')
        self.test_size = config['data_splits']['test_set_size']
        self.dev_size = config['data_splits']['dev_set_size']

    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits the data into train, dev and test sets.

        Args:
            data (pd.DataFrame): The input dataframe to be split.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The train, dev and test dataframes.
        """
        try:
            self.logger.info("Starting data split process...")
            # First split off the test set
            train_dev_data, test_data = train_test_split(
                data, test_size=self.test_size, random_state=42
            )
            test_dataset = mlflow.data.from_pandas(
                test_data, name='car_details_test_dataset', source='data_splitter.py'
            )
            # Then split the remaining data into train and dev sets
            train_data, dev_data = train_test_split(
                train_dev_data, test_size=self.dev_size, random_state=42
            )
            train_dataset = mlflow.data.from_pandas(
                train_data, name='car_details_train_dataset', source='data_splitter.py'
            )
            dev_dataset = mlflow.data.from_pandas(
                dev_data, name='car_details_dev_dataset', source='data_splitter.py'
            )
            self.logger.info("Data split completed successfully.")
            self.split_results = {
                "train": train_data,
                "dev": dev_data,
                "test": test_data
            }
            # log sizes to mlflow
            mlflow.log_params({
                'train_size' : len(train_data),
                'dev_size' : len(dev_data),
                'test_size' : len(test_data)
            })

            # log datasets to mlflow
            mlflow.log_input(train_dataset, context='training')
            mlflow.log_input(dev_dataset, context='development')
            mlflow.log_input(test_dataset, context='testing')

            return train_data, dev_data, test_data
        
        except Exception as e:
            self.logger.error(f"Error during data split: {e}")
            raise

    def validate_splits(self, original_df: pd.DataFrame) -> None:
        """Validates that the splits do not overlap and sizes are correct."""
        train_data = self.split_results['train']
        dev_data = self.split_results['dev']
        test_data = self.split_results['test']

        # Check for overlaps
        assert train_data.index.isin(dev_data.index).sum() == 0, "Train and Dev sets overlap!"
        assert train_data.index.isin(test_data.index).sum() == 0, "Train and Test sets overlap!"
        assert dev_data.index.isin(test_data.index).sum() == 0, "Dev and Test sets overlap!"

        # Check sizes
        total_size = len(train_data) + len(dev_data) + len(test_data)
        original_size = len(original_df)
        assert total_size == original_size, f"Total size {total_size} != original size {original_size}!"
        
        self.logger.info("Data splits validated successfully. No overlaps.")

    def save_splits(self, version: int = 1) -> None:
        """Saves the split datasets to CSV files.
        """
        train_data = self.split_results['train']
        dev_data = self.split_results['dev']
        test_data = self.split_results['test']

        train_path = self.ensure_paths(f'train_data_v{version}.csv')
        dev_path = self.ensure_paths(f"dev_data_v{version}.csv")
        test_path = self.ensure_paths(f"test_data_v{version}.csv")

        train_data.to_csv(train_path, index=False)
        dev_data.to_csv(dev_path, index=False)
        test_data.to_csv(test_path, index=False)

        mlflow.log_artifact(str(train_path))
        mlflow.log_artifact(str(dev_path))
        mlflow.log_artifact(str(test_path))

        self.logger.info(f"Train, Dev and Test datasets saved to {train_path.parent}")

    def ensure_paths(self, name: str) -> Path:
        """Ensure all directories are properly created
        
        Args:
            output_dir(str) : File directory
            name(str) : Name of file
            
        Returns:
            A filepath
        """
        output_dir = self.config['file_paths'].get('splits','data/splits/')

        file_path = Path(output_dir) / f"{name}"
        if file_path.is_dir():
            shutil.rmtree(file_path)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        return file_path
