"""Preprocessing pipeline for feature selection, encoding, and scaling.

This module orchestrates the feature selection, encoding, and scaling steps
of the preprocessing pipeline using the respective classes defined in their respective files.
Workflow:
1. Import necessary libraries.
2. Define a PreprocessingPipeline class to manage the workflow.
3. Initialize FeatureSelector, FeatureEncoder, and FeatureScaler classes.
4. Define a process method to execute the preprocessing steps sequentially.
5. Save processed data and log artifacts using mlflow."""

import pandas as pd
from typing import Dict, Any, List
import mlflow
from pathlib import Path
import numpy as np
import json
import shutil
import warnings
warnings.filterwarnings("ignore")
import sys
from dotenv import load_dotenv
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import (
    FeatureSelector,
    FeatureEncoder,
    FeatureScaler,
    DataCleaner, 
    DataSplitter,
    FeatureTransformer
)

from utils import setup_logger, read_yaml, Timer
from eda import DataLoader


class PreprocessingPipeline:
    """Class to manage the preprocessing pipeline."""

    def __init__(self, config_path: str | Path):
        """Initialize the preprocessing pipeline with configuration."""
        self.config = read_yaml(config_path)
        self.logger = setup_logger("PreprocessingPipeline","logs/")


    def execute(self):
        """Execute the preprocessing pipeline on the input DataFrame.
        """
        self.logger.info("="*80)
        self.logger.info("STARTING PREPROCESSING PIPELINE")
        self.logger.info("="*80)

        # load environment variables
        load_dotenv()

        TRACKING_URI = os.getenv("MLFLOW_TRACKER")
        if TRACKING_URI is None:
            raise ValueError("MLflow tracking URI not found in environment variables.")
        
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(experiment_name="Car Price Prediction Model - Preprocessing")

        try:
            with mlflow.start_run(run_name="Preprocessing_Pipeline") as run:
                mlflow.set_tag("stage", "preprocessing")

            # step 0: Data Loading
            with mlflow.start_run(run_name="Data_Loading", nested=True):
                with Timer("Data Loading", self.logger):
                    data_loader = DataLoader(self.config)
                    df = data_loader.load_dataset(self.config["file_paths"].get("raw_data", "data/raw/car_details.csv"))
                    mlflow.log_param('Original_dataset_shape', df.shape)
                

            # step 2: Data Splitter
            with mlflow.start_run(run_name="Data_Splitting", nested=True):
                with Timer("Data Splitting", self.logger):
                    splitter = DataSplitter(self.config)
                    train_df, test_df = splitter.split_data(df)
                    splitter.save_splits()


            # step 3: Data Cleaning
            with mlflow.start_run(run_name="Data_Cleaning", nested=True):
                with Timer("Data Cleaning", self.logger):
                    # clean train data
                    cleaner = DataCleaner(self.config)
                    train_df = cleaner.clean_data(train_df)
                    cleaner.save_cleaning_report('train')

                    # clean test data
                    test_df = cleaner.clean_data(test_df)
                    cleaner.save_cleaning_report('test')

                    self.logger.info(f'Train set shape after cleaning: {train_df.shape}')
                    self.logger.info(f'Test set shape after cleaning: {test_df.shape}')

                    mlflow.log_params({
                        'train_df_shape_after_cleaning' : train_df.shape,
                        'test_df_shape_after_cleaning' : test_df.shape
                    })


            # step 4: Feature Transformation
            with mlflow.start_run(run_name="Feature_Transformation", nested=True):
                with Timer("Feature Transformation", self.logger):
                    transformer = FeatureTransformer(self.config)
                    # transform train and test data
                    train_df = transformer.transform_features(train_df)
                    test_df = transformer.transform_features(test_df)
                    transformer.save_transformation_report()

                    self.logger.info(f'Train set shape after transformation: {train_df.shape}')
                    self.logger.info(f'Test set shape after transformation: {test_df.shape}')

                    mlflow.log_params({
                        'train_df_shape_after_transformation' : train_df.shape,
                        'test_df_shape_after_transformation' : test_df.shape
                    })


            # step 5: Feature Encoding
            with mlflow.start_run(run_name="Feature_Encoding", nested=True):
                with Timer("Feature Encoding", self.logger):
                    encoder = FeatureEncoder(self.config)
                    # fit_transform on train data and transform test data
                    train_df = encoder.fit_transform(train_df)
                    test_df = encoder.transform(test_df)

                    self.logger.info(f'Train set shape after encoding: {train_df.shape}')
                    self.logger.info(f'Test set shape after encoding: {test_df.shape}')

                    mlflow.log_params({
                        'train_df_shape_after_encoding' : train_df.shape,
                        'test_df_shape_after_encoding' : test_df.shape
                    })

            # step 6: Feature Scaling
            with mlflow.start_run(run_name="Feature_Scaling", nested=True):
                with Timer("Feature Scaling", self.logger):
                    scaler = FeatureScaler(self.config)
                    # fit_transform on train data and transform test data
                    numeric_columns_train = train_df.select_dtypes(include=[np.number]).columns.tolist() # numeric columns for train data
                    train_df = scaler.fit_transform(train_df, numeric_columns_train)

                    # numeric columns for test data
                    numeric_columns_test = test_df.select_dtypes(include=[np.number]).columns.tolist()
                    test_df = scaler.transform(test_df, numeric_columns_test)

                    mlflow.log_params({
                        'train_df_shape_after_scaling' : train_df.shape,
                        'test_df_shape_after_scaling' : test_df.shape
                    })

                    self.logger.info(f'Train set shape after scaling: {train_df.shape}')
                    self.logger.info(f'Test set shape after scaling: {test_df.shape}')


            # step 7: Feature Selection
            with mlflow.start_run(run_name="Feature_Selection", nested=True):
                with Timer("Feature Selection", self.logger):
                    selector = FeatureSelector(self.config)
                    # fit_transform on train data and transform test data
                    X_train = train_df.drop(columns=[self.config['target_variable']])
                    y_train = train_df[self.config['target_variable']]
                    selected_features = selector.fit_transform(X_train, y_train)
                    selector.save_selected_features(X_train)
                    
                    # Get selected feature names
                    selected_feature_names = selector.selected_features(X_train)

                    # Convert numpy array to DataFrame with selected feature names
                    selected_features = pd.DataFrame(selected_features, columns=selected_feature_names)
                    train_df = pd.concat([selected_features, y_train.reset_index(drop=True)], axis=1)


                    X_test = test_df.drop(columns=[self.config['target_variable']])
                    y_test = test_df[self.config['target_variable']]
                    selected_test_features = selector.transform(X_test)
                    selected_test_features = pd.DataFrame(selected_test_features, columns=selected_feature_names)
                    test_df = pd.concat([selected_test_features, y_test.reset_index(drop=True)], axis=1)

                    self.logger.info(f'Train set shape after feature selection: {train_df.shape}')
                    self.logger.info(f'Test set shape after feature selection: {test_df.shape}')



            # Save processed splits
            self.save_processed_splits(train_df,name='train', version=1, context="training")
            self.save_processed_splits(test_df,name='test', version=1, context="testing")

            mlflow.log_params({
                'final_train_shape' : train_df.shape,
                'final_test_shape' : test_df.shape
            })

            self.logger.info("="*80)
            self.logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*80)


        except Exception as e:
            self.logger.error(f'Error executing preprocessing pipeline: {e}')

    def save_processed_splits(self, df: pd.DataFrame, name: str, version: int = 1, context: str = "training") -> None:
        """Save the processed train, dev, and test splits to CSV files.

        Args:
            df (pd.DataFrame): The input DataFrame to be split and saved.
        """
        try:
            self.logger.info(f"Saving processed {name} split...")
            output_dir = self.config["file_paths"].get("processed_data", "data/processed/")

            df_path = Path(output_dir) / f"{name}_processed_v{version}.csv"

            if df_path.is_dir():
                shutil.rmtree(df_path)

            df_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(df_path, index=False)

            dataset = mlflow.data.from_pandas(
                df, name=f"processed_{name}_v{version}", source="preprocessing_pipeline.py"
            )

            mlflow.log_input(dataset, context=context)


            self.logger.info(f"Processed data splits saved at {output_dir}")

        except Exception as e:
            self.logger.error(f"Error saving processed splits: {e}")
            raise e


if __name__ == "__main__":
    pipeline = PreprocessingPipeline(config_path="config/preprocessing_config.yaml")
    pipeline.execute()
    

    