"""  
Core model training module with sklearn models
"""

import pandas as pd
import numpy as np
import time
import json
import joblib
from pathlib import Path
from utils import LoggerMixin
from typing import Dict, Any, List, Optional, Tuple
from sklearn.linear_model import Ridge 
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import shutil
import mlflow


class ModelTrainer(LoggerMixin):
    """
    Train regression models with cross-validation.
    
    Attributes:
        config: Configuration dictionary
        model: Trained model instance
        model_name: Name of the model
        training_time: Time taken to train
        cv_scores: Cross-validation scores
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelTrainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = self.setup_class_logger('model_trainer', config, 'logging')
        self.output_dir = Path(self.config['file_paths'].get('metrics_artifcts','artifacts/models/metrics'))
        self.model = None
        self.model_name = None
        self.training_time = 0.0
        self.cv_scores = None
        self.trained_models = {}
        self.results = {}
        self.eval_results = {}

    def validate_file_paths(self, report_path : str | Path) -> Path:
        """validate all filepaths and create new paths if needed"""
        if report_path.is_dir():
            shutil.rmtree(report_path)

        return report_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _get_model_instance(self, model_name: str, params: Dict[str, Any]):
        """
        Get model instance based on name.
        
        Args:
            model_name: Name of the model
            params: Model hyperparameters
            
        Returns:
            Model instance
        """
        model_map = {
            'Ridge': Ridge,
            'RandomForest': RandomForestRegressor,
            'XGBoost': XGBRegressor,
            'LightGBM': LGBMRegressor
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model_map[model_name](**params)
    
    def train(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            params: Model hyperparameters (optional)
            
        Returns:
            Trained model
        """
        self.logger.info(f"\nTraining {model_name}...")
        self.logger.info("-"*60)
        
        self.model_name = model_name
        
        # Get model parameters
        if params is None:
            params = self.config['baseline_models']['models'][model_name]['params']
        
        self.logger.info(f"Parameters: {params}")
        
        # Initialize model
        self.model = self._get_model_instance(model_name, params)
        
        # Train
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        self.logger.info(f"✓ Training completed in {self.training_time:.2f}s")
        
        return self.model
    
    def train_baseline_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all baseline models from config.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            scoring: Scoring metric for cross-validation
            
        Returns:
            Dictionary containing trained models and their CV scores
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("TRAINING BASELINE MODELS")
        self.logger.info("="*60)
        
        # Check if baseline models are enabled
        if not self.config.get('baseline_models', {}).get('enabled', False):
            self.logger.warning("Baseline models are disabled in config.")
            return {}
        
        models_config = self.config['baseline_models']['models']
        
        for model_name in models_config.keys():
            try:
                # Get params from config
                params = models_config[model_name]['params']
                
                # Train model
                trained_model = self.train(model_name, X_train, y_train, params)
                
                # Store results
                self.results[model_name] = {
                    'model': trained_model,
                    'params': params,
                    'training_time': self.training_time
                }


                
                self.trained_models[model_name] = trained_model
                
                self.logger.info(f"✓ {model_name} completed successfully\n")

                
            except Exception as e:
                self.logger.error(f"✗ Error training {model_name}: {str(e)}")
                raise e
            
        baseline_results_path = self.output_dir / "baseline_results.json"
        self.validate_file_paths(baseline_results_path)

        with open(baseline_results_path,'w') as f:
            json.dump(self.results, f, indent=4)

        # log to mlflow
        mlflow.log_artifact(baseline_results_path)
        
        self.logger.info("="*60)
        self.logger.info("BASELINE MODELS TRAINING COMPLETED")
        self.logger.info("="*60)
        
    
    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        X_dev: np.ndarray,
        y_dev: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate a single model on dev set.
        
        Args:
            model_name: Name of the model
            model: Trained model instance
            X_dev: Development features
            y_dev: Development labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_dev)
        
        # Calculate metrics
        mse = mean_squared_error(y_dev, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_dev, y_pred)
        mape = mean_absolute_percentage_error(y_dev, y_pred)
        
        metrics = {
            'model_name': model_name,
            'rmse': float(rmse),
            'mse': float(mse),
            'mae': float(mae),
            'mape': float(mape)
        }
        
        return metrics
    
    def evaluate_all_models(
        self,
        X_dev: np.ndarray,
        y_dev: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on dev set.
        
        Args:
            X_dev: Development features
            y_dev: Development labels
            
        Returns:
            Dictionary containing evaluation results for all models
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("EVALUATING MODELS ON DEV SET")
        self.logger.info("="*60)
        
        
        for model_name, model in self.trained_models.items():
            try:
                self.logger.info(f"\nEvaluating {model_name}...")
                metrics = self.evaluate_model(model_name, model, X_dev, y_dev)
                self.eval_results[model_name] = metrics
                
                self.logger.info(f"RMSE_{model_name}: {metrics['rmse']:.4f}")
                self.logger.info(f"MSE_{model_name}:  {metrics['mse']:.4f}")
                self.logger.info(f"MAE_{model_name}:  {metrics['mae']:.4f}")
                self.logger.info(f"MAPE_{model_name}: {metrics['mape']:.4f}")
                
            except Exception as e:
                self.logger.error(f"✗ Error evaluating {model_name}: {str(e)}")
                raise e
        
        return self.eval_results
    
    def get_top_models(self, top_k: int = 2) -> List[str]:
        """
        Select top K models based on dev RMSE.
        
        Args:
            top_k: Number of top models to select
            
        Returns:
            Tuple of (list of top model names, dict of their results)
        """
        self.logger.info("\n" + "="*60)
        self.logger.info(f"SELECTING TOP {top_k} MODELS")
        self.logger.info("="*60)
        
        # Sort models by RMSE
        sorted_models = sorted(
            self.eval_results.items(),
            key=lambda x: x[1].get('rmse', float('inf'))
        )
        
        top_models = sorted_models[:top_k]
        
        self.logger.info("\nRanking (sorted by RMSE):")
        for idx, (model_name, metrics) in enumerate(sorted_models, 1):
            self.logger.info(f"  {idx}. {model_name}: RMSE={metrics['rmse']:.4f}")
        
        self.logger.info(f"\n✓ Selected top {top_k} models:")
        top_model_dict = {}
        for idx, (model_name, metrics) in enumerate(top_models, 1):
            self.logger.info(f"  {idx}. {model_name} (RMSE: {metrics['rmse']:.4f})")
            top_model_dict[model_name] = {
                'model': self.trained_models[model_name],
                'metrics': metrics,
                'params': self.results[model_name]['params']
            }
        
        # save top model dictionary
        top_models_dict_path = self.output_dir / "top_models.dict.json"
        self.validate_file_paths(top_models_dict_path)

        with open(top_models_dict_path, 'w') as file:
            json.dump(top_model_dict, file, indent=4)

        mlflow.log_artifact(top_models_dict_path)

        top_model_names = [name for name, _ in top_models]
        
        return top_model_names
    
    def save_evaluation_results(self) -> str:
        """
        Save evaluation results to a JSON file.
        
        Args:
            filepath: Path to save the JSON file (optional)
            
        Returns:
            Path to the saved file
        """
        filepath = self.output_dir / f"evaluation_results.json"
        
        # Create directory if it doesn't exist
        self.validate_file_paths(filepath)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(self.eval_results, f, indent=4)
        
        self.logger.info(f"\n✓ Evaluation results saved to {filepath}")
        
        mlflow.log_artifact(filepath)
    
    def save_top_models(self, top_model_names: List[str], version: int = 1) -> Dict[str, str]:
        """
        Save top models for hyperparameter tuning.
        
        Args:
            top_model_names: List of top model names
            version: Version number for the saved models
            
        Returns:
            Dictionary with model names and their saved paths
        """
        
        saved_paths = {}
        
        for model_name in top_model_names:
            if model_name not in self.trained_models:
                self.logger.error(f"Model {model_name} not found in trained models.")
                continue
            
            model = self.trained_models[model_name]
            filepath = self.output_dir / f"{model_name}_v{version}.joblib"
            self.validate_file_paths(filepath)
            
            joblib.dump(model, filepath)
            saved_paths[model_name] = str(filepath)
            
            self.logger.info(f"✓ Saved {model_name} to {filepath}")
        
        return saved_paths
    
    def log_to_mlflow(
        self,
        eval_results_path: str,
        saved_model_paths: Dict[str, str]
    ) -> None:
        """
        Log evaluation results and top models to MLflow.
        
        Args:
            eval_results_path: Path to evaluation results JSON
            saved_model_paths: Dictionary of saved model paths
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("LOGGING TO MLFLOW")
        self.logger.info("="*60)
        
        
        # Log top models as artifacts
        for model_name, model_path in saved_model_paths.items():
            mlflow.log_artifact(model_path, artifact_path="top_models")
            self.logger.info(f"✓ Logged {model_name} model")
        
        # Log metrics for each model
        for model_name, metrics in self.eval_results.items():
            if 'error' not in metrics:
                mlflow.log_metrics({
                    f"{model_name}_rmse": metrics['rmse'],
                    f"{model_name}_mse": metrics['mse'],
                    f"{model_name}_mae": metrics['mae'],
                    f"{model_name}_mape": metrics['mape']
                })
        
        self.logger.info("✓ All artifacts and metrics logged to MLflow")
    
    def select_and_save_top_models(
        self,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
        top_k: int = 2,
        version: int = 1
    ) -> Dict[str, Any]:
        """
        Complete pipeline: evaluate, select, save, and log top models.
        
        Args:
            X_dev: Development features
            y_dev: Development labels
            top_k: Number of top models to select
            version: Version number for saved models
            
        Returns:
            Dictionary containing top model info, paths, and metrics
        """
        # Evaluate all models
        self.evaluate_all_models(X_dev, y_dev)
        
        # Get top models
        top_model_names = self.get_top_models(top_k)
        
        # Save evaluation results
        eval_results_path = self.save_evaluation_results()
        
        # Save top models
        saved_model_paths = self.save_top_models(top_model_names, version)
        
        # Log to MLflow
        self.log_to_mlflow(eval_results_path, saved_model_paths)
        
        self.logger.info("\n" + "="*60)
        self.logger.info("MODEL SELECTION AND SAVING COMPLETED")
        self.logger.info("="*60)
        
        return {
            'top_models': top_model_names,
            'eval_results_path': eval_results_path,
            'saved_model_paths': saved_model_paths,
            'all_eval_results': self.eval_results
        }