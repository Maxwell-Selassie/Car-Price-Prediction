"""Evaluation pipeline Module"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import mlflow 

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import LoggerMixin, Timer
from model_evaluation import LoadData, LoadModel, RegressionMetrics,RunPredictions

class ModelEvaluationPipeline(LoggerMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.logger = self.setup_class_logger("ModelEvaluation",config,"logging")
        self.load_data = LoadData(config)
        self.load_model = LoadModel(config)
        self.run_predictions = RunPredictions(config)
        self.regression_metrics = RegressionMetrics(config)
        self.pipeline_results = {}
        self.y_pred = None
        self.x_test = None
        self.y_test = None 
        self.best_model = None 
        self.eval_metrics = {}

    def execute(self):
        """Run model evaluation"""
        self.logger.info(f"="*50)
        self.logger.info(f"MODEL EVALUATION STARTED")
        self.logger.info("="*50) 

        load_dotenv() 

        TRACKING_URI = os.getenv("MLFLOW_TRACKER")
        if TRACKING_URI is None:
            raise ValueError("MLflow tracking URI not found in environment variables.")
        
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(experiment_name="Car Price Prediction Model - Model-Evaluation")

        with mlflow.start_run(run_name="ModelEvaluation_pipeline") as run:
            mlflow.set_tag("stage", "evaluation")

        # step 1: Data Loading
        with mlflow.start_run(run_name="Data Loaiding", nested=True):
            with Timer("Load_data", self.logger):
                self.logger.info("="*50)
                self.logger.info("LOADING TEST DATA")
                self.logger.info("="*50)

                try:
                    self.load_data.load_test_data()

                    self.x_test = self.load_data.X_test
                    self.y_test = self.load_data.y_test

                    data_info = {
                        "test_samples" : len(self.x_test),
                        "n_features" : len(self.x_test.columns),
                        "feature_names" : self.x_test.columns.tolist()
                    }

                    self.logger.info(f"Data loaded successfully")
                    self.logger.info("Test samples: ",data_info['test_samples'])
                    self.logger.info("Number of features: ",data_info["n_features"])

                    self.pipeline_results["data_loading"] = data_info
                
                except Exception as e:
                    self.logger.error(f"Error in step 1 - Data Loading: {e}")
                    raise 
            
        # step 2: Load Model
        with mlflow.start_run(run_name="Model_Loading", nested=True):
            with Timer("Model_Loading", self.logger):
                self.logger.info("="*50)
                self.logger.info("STEP 1: LOADING MODEL")
                self.logger.info("="*50)

                try: 
                    # load best model from mlflow
                    self.best_model = self.load_model.load_model()

                    self.logger.info(f"Best Model successfully loaded from mlflow")

                except Exception as e:
                    self.logger.error(f"Error in step 2 - Model loading failed: {e}")
                    raise 

        # step 3: Run Predictions
        with mlflow.start_run(run_name="Run_Predictions", nested=True):
            with Timer("Run_Predictions", self.logger):

                self.logger.info("="*50)
                self.logger.info("STEP 1: RUNNING PREDICTIONS")
                self.logger.info("="*50)

                try: 
                    self.y_pred = self.run_predictions.predictions(self.best_model, self.x_test)

                    self.logger.info(f"Predictions run successfully")

                except Exception as e:
                    self.logger.error(f"Error in step 3 - Predictions run successfully on test data")
                    raise 

        # step 4: evaluate model with regression metrics
        with mlflow.start_run(run_name="Regression Metrics", nested=True):
            with Timer("Regression metrics", self.logger):

                self.logger.info("="*50)
                self.logger.info("STEP 1: REGRESSION METRICS")
                self.logger.info("="*50)

                try:
                    self.eval_metrics = self.regression_metrics.metrics(self.y_test, self.y_pred)

                    mse = self.eval_metrics['mse']
                    rmse = self.eval_metrics['rmse']
                    mae = self.eval_metrics['mae']
                    mape = self.eval_metrics['mape']
                    r2_score = self.eval_metrics['r2_score']

                    mlflow.log_params({
                        "Test mse" : mse,
                        "Test rmse" : rmse,
                        "Test mae" : mae,
                        "Test mape" : mape,
                        "Test r2_score" : r2_score
                    })

                    self.logger.info(f"Test MSE: {mse:.4f}")
                    self.logger.info(f"Test RMSE: {rmse:.4f}")
                    self.logger.info(f"Test MAE: {mae:.4f}")
                    self.logger.info(f"Test MAPE: {mape:.4f}")
                    self.logger.info(f"Test R2_score: {r2_score:.4f}")

                except Exception as e:
                    self.logger.error(f"Error in step 4 - Error evaluating model using regression metrics")
                    raise 
        
        self.logger.info(f"="*50)
        self.logger.info(f"MODEL EVALUATION COMPLETED")
        self.logger.info(f"="*50)

def main(config_path: str = 'config/evaluation_config.yaml'):
    """
    Main entry point for the pipeline.
    
    Args:
        config_path: Path to configuration YAML file
    """
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run pipeline
    pipeline = ModelEvaluationPipeline(config)
    pipeline.execute()
    



if __name__ == "__main__":
    # Run the pipeline
    main()