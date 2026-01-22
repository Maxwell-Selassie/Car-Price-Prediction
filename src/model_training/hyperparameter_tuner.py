"""Hyperparameter using optuna for regression models.

Workflow:
1. Import necessary libraries and define HyperParameterTuner class.
2. Implement objective function for optimization.
3. Set up and run Optuna study.
4. Log and visualize results to mlflow."""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from utils import LoggerMixin
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_slice, plot_param_importances
import mlflow
class HyperParameterTuner(LoggerMixin):
    """
    Hyperparameter tuning with Optuna.
    
    Attributes:
        config: Configuration dictionary
        study: Optuna study object
        best_params: Best hyperparameters found
        best_score: Best score achieved
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = self.setup_class_logger("HyperParameterTuner",config,"logging")

        self.study = None
        self.best_params = None
        self.best_score = None 

    def objective(self, trial: optuna.trial.Trial, X: pd.DataFrame, y: pd.Series, model_name: str) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            X: Feature DataFrame
            y: Target Series
            model_name: Name of the regression model
        Returns:
            Mean cross-validation score
        """
        if model_name == "RandomForest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
            model = RandomForestRegressor(**params, random_state=42)
        
        elif model_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
            }
            model = XGBRegressor(**params, random_state=42)
        
        elif model_name == "LightGBM":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', -1, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
            model = LGBMRegressor(**params, random_state=42)
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        mean_score = np.mean(scores)
        
        return mean_score
    
    def tune(self, X: pd.DataFrame, y: pd.Series, model_name: str) -> None:
        """
        Run hyperparameter tuning.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_name: Name of the regression model
        """
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=3)
        
        self.study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        
        self.logger.info(f"Starting hyperparameter tuning for {model_name}...")
        
        self.study.optimize(lambda trial: self.objective(trial, X, y, model_name), n_trials=self.config.get('n_trials', 50))
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        self.logger.info(f"Best score: {self.best_score}")
        self.logger.info(f"Best parameters: {self.best_params}")

    def plot_resutls(self) -> None:
        """
        Generate and log optimization plots to mlflow.
        """
        if self.study is None:
            self.logger.error("Study has not been created. Run tune() first.")
            return
        
        fig1 = plot_optimization_history(self.study)
        fig2 = plot_slice(self.study)
        fig3 = plot_param_importances(self.study)
        
        mlflow.log_figure(fig1, "optimization_history.png")
        mlflow.log_figure(fig2, "slice_plot.png")
        mlflow.log_figure(fig3, "param_importances.png")
        
        self.logger.info("Optimization plots logged to mlflow.")

    