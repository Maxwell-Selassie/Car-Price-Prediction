"""
Hyperparameter tuning module using Optuna for model optimization.
"""

import numpy as np
import pandas as pd
import time
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.trial import Trial
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import mlflow
import matplotlib.pyplot as plt
from utils import LoggerMixin
import plotly.graph_objects as go

class HyperparameterTuner(LoggerMixin):
    """
    Hyperparameter tuning using Optuna for regression models.
    
    Attributes:
        config: Configuration dictionary
        study: Optuna study object
        best_params: Best hyperparameters found
        best_model: Best model after retraining
        tuning_results: Results of tuning process
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HyperparameterTuner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = self.setup_class_logger('hyperparameter_tuner', config, 'logging')
        self.study = None
        self.best_params = {}
        self.best_model = None
        self.best_model_name = None
        self.tuning_results = {}
        self.trained_models = {}
        self.eval_results = {}
    
    def _get_sampler(self, sampler_name: str):
        """
        Get Optuna sampler.
        
        Args:
            sampler_name: Name of the sampler
            
        Returns:
            Optuna sampler instance
        """
        samplers = {
            'TPESampler': TPESampler(seed=self.config['hyperparameter_tuning']['random_state']),
            'RandomSampler': RandomSampler(seed=self.config['hyperparameter_tuning']['random_state'])
        }
        
        if sampler_name not in samplers:
            self.logger.warning(f"Unknown sampler {sampler_name}, using TPESampler")
            return samplers['TPESampler']
        
        return samplers[sampler_name]
    
    def _get_pruner(self, pruner_name: str):
        """
        Get Optuna pruner.
        
        Args:
            pruner_name: Name of the pruner
            
        Returns:
            Optuna pruner instance
        """
        pruners = {
            'MedianPruner': MedianPruner(),
            'SuccessiveHalvingPruner': SuccessiveHalvingPruner()
        }
        
        if pruner_name not in pruners:
            self.logger.warning(f"Unknown pruner {pruner_name}, using MedianPruner")
            return pruners['MedianPruner']
        
        return pruners[pruner_name]
    
    def _parse_search_space(self, param_name: str, param_config: list, trial: Trial) -> Any:
        """
        Parse and apply search space configuration.
        
        Args:
            param_name: Parameter name
            param_config: Parameter configuration [type, *args]
            trial: Optuna trial object
            
        Returns:
            Parameter value suggested by trial
        """
        param_type = param_config[0]
        
        if param_type == 'int':
            return trial.suggest_int(param_name, param_config[1], param_config[2])
        elif param_type == 'float' or param_type == 'uniform':
            return trial.suggest_float(param_name, param_config[1], param_config[2])
        elif param_type == 'log_uniform':
            return trial.suggest_float(param_name, param_config[1], param_config[2], log=True)
        elif param_type == 'categorical':
            return trial.suggest_categorical(param_name, param_config[1])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    def _create_objective(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray):
        """
        Create objective function for Optuna optimization.
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Objective function
        """
        search_space = self.config['hyperparameter_tuning']['search_spaces'][model_name]
        
        def objective(trial: Trial) -> float:
            # Suggest hyperparameters
            params = {}
            for param_name, param_config in search_space.items():
                params[param_name] = self._parse_search_space(param_name, param_config, trial)
            
            try:
                # Create model instance
                if model_name == 'Ridge':
                    model = Ridge(**params)
                elif model_name == 'RandomForest':
                    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                elif model_name == 'XGBoost':
                    model = XGBRegressor(**params, random_state=42, verbosity=0)
                elif model_name == 'LightGBM':
                    model = LGBMRegressor(**params, random_state=42, verbose=-1)
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
                # Train on training data
                cv_scores = cross_val_score(
                    estimator=model, X=X_train,
                    y=y_train, cv=KFold(5), 
                    scoring='neg_root_mean_sqaured_error', n_jobs=-1
                )
                
                return -cv_scores.mean()
            
            except Exception as e:
                self.logger.warning(f"Trial failed with error: {str(e)}")
                raise e
        
        return objective
    
    def tune_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters for a single model.
        
        Args:
            model_name: Name of the model to tune
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary containing best parameters and study info
        """
        self.logger.info("\n" + "="*60)
        self.logger.info(f"TUNING HYPERPARAMETERS FOR {model_name}")
        self.logger.info("="*60)
        
        hp_config = self.config['hyperparameter_tuning']
        
        # Create objective function
        objective = self._create_objective(model_name, X_train, y_train)
        
        # Create study
        sampler = self._get_sampler(hp_config['sampler'])
        pruner = self._get_pruner(hp_config['pruner'])
        
        self.study = optuna.create_study(
            direction=hp_config['direction'],
            sampler=sampler,
            pruner=pruner
        )
        
        # Optimize
        start_time = time.time()
        self.study.optimize(
            objective,
            n_trials=hp_config['n_trials'],
            show_progress_bar=True
        )
        tuning_time = time.time() - start_time
        
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        self.logger.info(f"\n✓ Tuning completed in {tuning_time:.2f}s")
        self.logger.info(f"Best RMSE: {best_value:.4f}")
        self.logger.info(f"Best parameters: {best_params}")
        
        return {
            'model_name': model_name,
            'best_params': best_params,
            'best_rmse': float(best_value),
            'n_trials': hp_config['n_trials'],
            'tuning_time': tuning_time,
            'study': self.study
        }
    
    def retrain_with_best_params(
        self,
        model_name: str,
        best_params: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Any:
        """
        Retrain model with best hyperparameters.
        
        Args:
            model_name: Name of the model
            best_params: Best hyperparameters
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model instance
        """
        self.logger.info(f"\nRetraining {model_name} with best parameters...")
        
        try:
            if model_name == 'Ridge':
                model = Ridge(**best_params)
            elif model_name == 'RandomForest':
                model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
            elif model_name == 'XGBoost':
                model = XGBRegressor(**best_params, random_state=42, verbosity=0)
            elif model_name == 'LightGBM':
                model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            self.trained_models[model_name] = model
            self.logger.info(f"✓ {model_name} retrained in {training_time:.2f}s")
            
            return model
        
        except Exception as e:
            self.logger.error(f"✗ Error retraining {model_name}: {str(e)}")
            raise
    
    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        X_dev: np.ndarray,
        y_dev: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on dev set.
        
        Args:
            model_name: Name of the model
            model: Trained model instance
            X_dev: dev features
            y_dev: dev labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_dev)
        
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
        
        self.logger.info(f"\n{model_name} Evaluation Results:")
        self.logger.info(f"RMSE_{model_name}: {rmse:.4f}")
        self.logger.info(f"MSE_{model_name}:  {mse:.4f}")
        self.logger.info(f"MAE_{model_name}:  {mae:.4f}")
        self.logger.info(f"MAPE_{model_name}: {mape:.4f}")
        
        return metrics
    
    def plot_optimization_results(self, model_name: str, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Plot Optuna optimization results.
        
        Args:
            model_name: Name of the model
            output_dir: Directory to save plots (optional)
            
        Returns:
            Dictionary of plot file paths
        """
        if not self.config.get('plotting', {}).get('enabled', False):
            self.logger.info("Plotting is disabled in config")
            return {}
        
        if output_dir is None:
            output_dir = self.config['file_paths'].get('figures_artifacts','artifacts/models/figures/')
        
        output_path = Path(output_dir) / model_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_paths = {}
        
        try:
            # Plot optimization history
            if self.config.get('plotting', {}).get('optuna', {}).get('plot_optimization_history', False):
                fig = optuna.visualization.plot_optimization_history(self.study).to_plotly_json()
                history_path = output_path / "optimization_history.html"

                go.Figure(fig).write_html(history_path)
                plot_paths['optimization_history'] = str(history_path)
                self.logger.info(f"✓ Saved optimization history plot")
            
            # Plot slice
            if self.config.get('plotting', {}).get('optuna', {}).get('plot_slice', False):
                fig = optuna.visualization.plot_slice(self.study).to_plotly_json()
                slice_path = output_path / "slice.html"
                go.Figure(fig).write_html(slice_path)
                plot_paths['slice'] = str(slice_path)
                self.logger.info(f"✓ Saved slice plot")
            
            # Plot parameter importances
            if self.config.get('plotting', {}).get('optuna', {}).get('plot_param_importances', False):
                fig = optuna.visualization.plot_param_importances(self.study).to_plotly_json()
                importance_path = output_path / "param_importances.html"
                go.Figure(fig).write_html(importance_path)
                plot_paths['param_importances'] = str(importance_path)
                self.logger.info(f"✓ Saved parameter importances plot")
        
        except Exception as e:
            self.logger.warning(f"Error generating plots: {str(e)}")
        
        return plot_paths
    
    def save_tuning_results(self, output_dir: Optional[str] = None) -> str:
        """
        Save tuning results to JSON file.
        
        Args:
            output_dir: Directory to save results (optional)
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = self.config['file_paths'].get('metrics_artifacts','artifacts/models/metrics/')
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / 'tuning_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.tuning_results, f, indent=4)
        
        self.logger.info(f"✓ Tuning results saved to {results_file}")
        
        return str(results_file)
    
    def save_best_model(self, model_name: str, output_dir: Optional[str] = None) -> str:
        """
        Save best model with 'production' alias.
        
        Args:
            model_name: Name of the model
            output_dir: Directory to save model (optional)
            
        Returns:
            Path to saved model
        """
        if output_dir is None:
            output_dir = self.config['file_paths'].get('model_artifacts','artifacts/models/')
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_file = output_path / f'{model_name}_production.joblib'
        
        if model_name not in self.trained_models:
            self.logger.error(f"Model {model_name} not found in trained models")
            return None
        
        joblib.dump(self.trained_models[model_name], model_file)
        self.logger.info(f"✓ Best model saved to {model_file} with alias 'production'")
        
        return str(model_file)
    
    def log_to_mlflow(
        self,
        model_name: str,
        best_params: Dict[str, Any],
        eval_metrics: Dict[str, float],
        model_path: str,
        plot_paths: Dict[str, str]
    ) -> None:
        """
        Log best model and results to MLflow.
        
        Args:
            model_name: Name of the model
            best_params: Best hyperparameters
            eval_metrics: Evaluation metrics
            model_path: Path to saved model
            plot_paths: Dictionary of plot paths
        """
        self.logger.info(f"\nLogging {model_name} to MLflow...")
        
        # Log parameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"{model_name}_{param_name}", param_value)
        
        # Log metrics
        for metric_name, metric_value in eval_metrics.items():
            if metric_name != 'model_name':
                mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)
        
        # Log model
        mlflow.log_artifact(model_path, artifact_path=f"{model_name}/production")
        
        # Log plots
        for plot_name, plot_path in plot_paths.items():
            mlflow.log_artifact(plot_path, artifact_path=f"{model_name}/plots")
        
        # Log tuning results
        tuning_results_path = self.save_tuning_results()
        mlflow.log_artifact(tuning_results_path, artifact_path=f"{model_name}/tuning")
        
        self.logger.info(f"✓ {model_name} logged to MLflow")
    
    def tune_and_retrain_models(
        self,
        model_names: list,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Complete pipeline: tune, retrain, evaluate, and save best model.
        
        Args:
            model_names: List of model names to tune
            X_train: Training features
            y_train: Training labels
            X_dev: Validation features
            y_dev: Validation labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing best model name and its details
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("HYPERPARAMETER TUNING AND RETRAINING")
        self.logger.info("="*60)
        
        tuning_results = {}
        eval_results = {}
        
        # Tune each model
        for model_name in model_names:
            try:
                # Tune hyperparameters
                tune_result = self.tune_model(model_name, X_train, y_train)
                
                # Retrain with best params
                best_model = self.retrain_with_best_params(
                    model_name,
                    tune_result['best_params'],
                    X_train,
                    y_train
                )
                
                # Evaluate
                metrics = self.evaluate_model(model_name, best_model, X_dev, y_dev)
                
                tuning_results[model_name] = tune_result
                eval_results[model_name] = metrics
                
                # Plot optimization results
                plot_paths = self.plot_optimization_results(model_name)
                
                # Save model
                model_path = self.save_best_model(model_name)
                
                # Log to MLflow
                with mlflow.start_run(run_name=f"{model_name}_tuning", nested=True):
                    self.log_to_mlflow(model_name, tune_result['best_params'], metrics, model_path, plot_paths)
                
            except Exception as e:
                self.logger.error(f"✗ Error tuning {model_name}: {str(e)}")
        
        # Select best model based on test RMSE
        self.logger.info("\n" + "="*60)
        self.logger.info("SELECTING BEST MODEL")
        self.logger.info("="*60)
        
        best_model_name = min(eval_results, key=lambda x: eval_results[x]['rmse'])
        best_metrics = eval_results[best_model_name]
        
        self.logger.info(f"\n✓ Best Model: {best_model_name}")
        self.logger.info(f"  RMSE: {best_metrics['rmse']:.4f}")
        self.logger.info(f"  MSE:  {best_metrics['mse']:.4f}")
        self.logger.info(f"  MAE:  {best_metrics['mae']:.4f}")
        self.logger.info(f"  MAPE: {best_metrics['mape']:.4f}")
        
        self.best_model_name = best_model_name
        self.best_model = self.trained_models[best_model_name]
        self.best_params = tuning_results[best_model_name]['best_params']
        
        mlflow.log_params({
            'best_model_name': best_model_name,
            'best_model': self.best_model,
            'best_params': self.best_params,
            'best_metrics': best_metrics,
            'tuning_results': tuning_results,
            'eval_results': eval_results
        })
    
