"""
Model Training Pipeline Module

This module orchestrates the complete machine learning model training pipeline:
1. Data loading
2. Baseline model training and evaluation
3. Hyperparameter tuning with cross-validation
4. Best model selection and registration to MLflow
5. Comprehensive logging and artifact management
"""

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import sys
import os
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from model_training import DataLoader, HyperparameterTuner, ModelTrainer
from utils import LoggerMixin, Timer


class ModelTrainingPipeline(LoggerMixin):
    """
    Complete ML pipeline for training, tuning, and registering models.
    
    Attributes:
        config: Configuration dictionary
        data_loader: DataLoader instance
        model_trainer: ModelTrainer instance
        hyperparameter_tuner: HyperparameterTuner instance
        best_model_info: Information about the best model
        pipeline_results: Comprehensive pipeline results
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelTrainingPipeline.
        
        Args:
            config: Configuration dictionary from YAML
        """
        self.config = config
        self.logger = self.setup_class_logger('model_training_pipeline', config, 'logging')
        
        self.data_loader = DataLoader(config)
        self.model_trainer = ModelTrainer(config)
        self.hyperparameter_tuner = HyperparameterTuner(config)
        
        self.best_model_info = None
        self.pipeline_results = {}
        self.X_train = None
        self.y_train = None
        self.top_model_names = []
        self.best_model = None

    def execute(self):
        """Execute the preprocessing pipeline on the input DataFrame.
        """
        self.logger.info("="*80)
        self.logger.info("STARTING TRAINING PIPELINE")
        self.logger.info("="*80)

        # load environment variables
        load_dotenv()

        TRACKING_URI = os.getenv("MLFLOW_TRACKER")
        if TRACKING_URI is None:
            raise ValueError("MLflow tracking URI not found in environment variables.")
        
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(experiment_name="Car Price Prediction Model - Model-Training")

        try: 
            with mlflow.start_run(run_name="ModelTraining_Pipeline") as run:
                mlflow.set_tag('stage','training')

            # step 0: Data Loading
            with mlflow.start_run(run_name='Data_Loading', nested=True):
                with Timer('Data_Loading', self.logger):
                    self.logger.info("\n" + "="*70)
                    self.logger.info("STEP 1: LOADING DATA")
                    self.logger.info("="*70)

                    try:
                        self.data_loader.load()
                        self.data_loader.get_metadata_and_save_metadata()
                        
                        self.X_train = self.data_loader.X_train
                        self.y_train = self.data_loader.y_train
                        
                        data_info = {
                            'train_samples': len(self.X_train),
                            'n_features': self.data_loader.n_features,
                            'feature_names': self.data_loader.feature_names
                        }
                        
                        self.logger.info(f"✓ Data loaded successfully")
                        self.logger.info(f"Train samples: {data_info['train_samples']}")
                        self.logger.info(f"Number of features: {data_info['n_features']}")
                        
                        self.pipeline_results['data_loading'] = data_info
                        
                
                    except Exception as e:
                        self.logger.error(f"✗ Error in Step 1 - Data Loading: {str(e)}")
                        raise

            # step 1: Model Trainer
            with mlflow.start_run(run_name='Model_Trainer', nested=True):
                with Timer('Model_Training', self.logger):
                    self.logger.info("\n" + "="*70)
                    self.logger.info("STEP 2: TRAINING BASELINE MODELS")
                    self.logger.info("="*70)
                    
                    try:
                        # Train all baseline models on training data
                        self.model_trainer.train_baseline_models(self.X_train, self.y_train)
                        
                        model_training_results = self.model_trainer.select_and_save_top_models(self.X_train, self.y_train)
                        
                        baseline_info = {
                            'top_models': model_training_results['top_models'],
                            'eval_results': self.model_trainer.eval_results,
                            'saved_paths': model_training_results['saved_model_paths']
                        }
                                    
                        self.logger.info(f"✓ Baseline model training completed")
                        self.logger.info(f"Top models selected: {model_training_results['top_models']}")
                        
                        self.pipeline_results['baseline_training'] = baseline_info
                        
                        self.top_model_names = baseline_info['top_models']
                    
                    except Exception as e:
                        self.logger.error(f"✗ Error in Step 2 - Baseline Model Training: {str(e)}")
                        raise

            # step 2: Hyperparameter Tuning
            with mlflow.start_run(run_name='Hyperparameter_Tuner', nested=True):
                with Timer('Hyperparameter_tuning', self.logger):
                    self.logger.info("\n" + "="*70)
                    self.logger.info("STEP 3: HYPERPARAMETER TUNING WITH OPTUNA (CV as Validation)")
                    self.logger.info("="*70)
                    try: 
                        hyperparameter_tuning_results = self.hyperparameter_tuner.tune_and_retrain_models(
                                    self.top_model_names, self.X_train, self.y_train
                        )

                        self.best_model = hyperparameter_tuning_results['best_model']

                        hp_info = {
                            'tuning_results' : hyperparameter_tuning_results['tuning_results'],
                            'eval_results' : hyperparameter_tuning_results['eval_results']
                        }

                        self.best_model_info = {
                            'model_name' : hyperparameter_tuning_results['best_model_name'],
                            'best_params' : hyperparameter_tuning_results['best_params'],
                            'best_metrics' : hyperparameter_tuning_results['best_metrics'],
                            'tuning_info' : hyperparameter_tuning_results['tuning_results'][hyperparameter_tuning_results['best_model_name']]
                        }

                        self.pipeline_results['hyperparameter_tuning'] = hp_info
                        self.logger.info(f"✓ Hyperparameter tuning completed")
                    except Exception as e:
                        self.logger.error(f"✗ Error in Step 3 - Hyperparameter Tuning: {str(e)}")
                        raise

            # step 3: register best model
            with Timer('Register best model', self.logger):
                self.logger.info("\n" + "="*70)
                self.logger.info("STEP 5: REGISTERING MODEL TO MLFLOW (DEFENSIVE)")
                self.logger.info("="*70)
                
                try:
                    if self.best_model_info is None:
                        raise ValueError("No best model found. Please run previous steps first.")
                    
                    model_name = self.best_model_info['model_name']
                    new_model_rmse = self.best_model_info['best_metrics']['rmse']
                    
                    # Get MLflow client
                    client = mlflow.MlflowClient()
                    
                    # Check if a model with 'production' alias already exists
                    production_alias_exists = False
                    existing_model_uri = None
                    existing_model_name = None
                    existing_model_rmse = None
                    
                    self.logger.info("\nChecking for existing 'production' model...")
                    
                    try:
                        # Search for all registered models
                        registered_models = client.search_registered_models()
                        
                        for reg_model in registered_models:
                            # Check if this model has a 'production' alias
                            aliases = [alias.alias for alias in reg_model.aliases]
                            if 'production' in aliases:
                                production_alias_exists = True
                                existing_model_name = reg_model.name
                                
                                # Get the version with production alias
                                for alias in reg_model.aliases:
                                    if alias.alias == 'production':
                                        existing_model_version = alias.version
                                        existing_model_uri = f"models:/{existing_model_name}@production"
                                        
                                        self.logger.info(f"✓ Found existing 'production' model: {existing_model_name}")
                                        self.logger.info(f"  Version: {existing_model_version}")
                                        
                                        # Try to load and evaluate existing model
                                        try:
                                            existing_model = mlflow.pyfunc.load_model(existing_model_uri)
                                            y_existing_pred = existing_model.predict(
                                                pd.DataFrame(self.X_train, columns=self.data_loader.feature_names)
                                            )
                                            
                                            from sklearn.metrics import mean_squared_error
                                            existing_model_mse = mean_squared_error(self.y_train, y_existing_pred)
                                            existing_model_rmse = np.sqrt(existing_model_mse)
                                            
                                            self.logger.info(f"  Existing Model RMSE: {existing_model_rmse:.4f}")
                                            
                                        except Exception as e:
                                            self.logger.warning(f"Could not load existing model for comparison: {str(e)}")
                                            self.logger.info("Proceeding with new model registration anyway...")
                                            existing_model_rmse = None
                                        
                                        break
                                break
                    
                    except Exception as e:
                        self.logger.info(f"No existing production model found or error checking: {str(e)}")
                        production_alias_exists = False
                    
                    # If production model exists, check performance threshold
                    if production_alias_exists and existing_model_rmse is not None:
                        self.logger.info(f"\nComparing model performance...")
                        self.logger.info(f"  New Model RMSE: {new_model_rmse:.4f}")
                        self.logger.info(f"  Existing Model RMSE: {existing_model_rmse:.4f}")
                        
                        # Calculate percentage improvement
                        # Positive percentage means new model is better (lower RMSE)
                        improvement_percentage = ((existing_model_rmse - new_model_rmse) / existing_model_rmse) * 100
                        threshold = 0.1  # 0.1% improvement required
                        
                        self.logger.info(f"  Improvement: {improvement_percentage:.4f}%")
                        self.logger.info(f"  Threshold: {threshold}%")
                        
                        if improvement_percentage < threshold:
                            self.logger.warning(f"\n✗ New model does NOT meet performance threshold")
                            self.logger.warning(f"  New model needs at least {threshold}% improvement but only achieved {improvement_percentage:.4f}%")
                            self.logger.warning(f"  Keeping existing 'production' model: {existing_model_name}")
                            
                            self.pipeline_results['model_registry'] = {
                                'status': 'SKIPPED',
                                'reason': f'New model did not meet {threshold}% improvement threshold',
                                'new_model_name': model_name,
                                'new_model_rmse': new_model_rmse,
                                'existing_model_name': existing_model_name,
                                'existing_model_rmse': existing_model_rmse,
                                'improvement_percentage': improvement_percentage,
                                'uri': existing_model_uri
                            }
                            
                            self.logger.info("\nRegistration skipped. Production model unchanged.")
                            return existing_model_uri
                        
                        else:
                            self.logger.info(f"\n✓ New model MEETS performance threshold")
                            self.logger.info(f"  Improvement: {improvement_percentage:.4f}% (required: {threshold}%)")
                            self.logger.info(f"  Will replace existing production model with new model")
                    
                    # Prepare input example (first 5 rows)
                    X_example = pd.DataFrame(
                        self.X_train[:5],
                        columns=self.data_loader.feature_names
                    )
                    
                    # Make predictions on example for signature
                    y_example = self.best_model.predict(X_example)
                    
                    # Infer signature
                    signature = infer_signature(X_example, y_example)
                    
                    # Log best model parameters and metrics to MLflow
                    mlflow.log_params({
                        f"best_model_{key}": str(value) 
                        for key, value in self.best_model_info['best_params'].items()
                    })
                    
                    mlflow.log_metrics({
                        f"best_model_{key}": value 
                        for key, value in self.best_model_info['best_metrics'].items()
                        if key != 'model_name'
                    })
                    
                    # Log additional tuning metadata
                    mlflow.log_params({
                        'best_model_name': model_name,
                        'n_tuning_trials': str(self.best_model_info['tuning_info']['n_trials']),
                        'tuning_time_seconds': str(self.best_model_info['tuning_info']['tuning_time']),
                        'cv_best_rmse': str(self.best_model_info['tuning_info']['best_rmse'])
                    })
                    
                    # Log model using sklearn flavor with signature and input example
                    mlflow.sklearn.log_model(
                        sk_model=self.best_model,
                        name=f"{model_name}_production",
                        signature=signature,
                        input_example=X_example,
                        registered_model_name=f"{model_name}_model"
                    )
                    
                    # Get the latest version of the registered model
                    model_versions = client.search_model_versions(f"name='{model_name}_model'")
                    latest_version = max(model_versions, key=lambda x: int(x.version)).version
                    
                    # If production alias exists on a different model, remove it first
                    if production_alias_exists and existing_model_name and existing_model_name != f"{model_name}_model":
                        self.logger.info(f"\nRemoving 'production' alias from old model: {existing_model_name}")
                        try:
                            client.delete_registered_model_alias(
                                name=existing_model_name,
                                alias="production"
                            )
                            self.logger.info(f"✓ Removed 'production' alias from {existing_model_name}")
                        except Exception as e:
                            self.logger.warning(f"Could not remove alias from old model: {str(e)}")
                    
                    # Set alias to 'production' on new model
                    client.set_registered_model_alias(
                        name=f"{model_name}_model",
                        alias="production",
                        version=latest_version
                    )
                    
                    model_uri = f"models:/{model_name}_model@production"
                    
                    self.logger.info(f"\n✓ Model registered to MLflow Registry")
                    self.logger.info(f"  Model Name: {model_name}_model")
                    self.logger.info(f"  Version: {latest_version}")
                    self.logger.info(f"  Alias: production")
                    self.logger.info(f"  URI: {model_uri}")
                    
                    if production_alias_exists and existing_model_rmse is not None:
                        self.logger.info(f"Replaced previous production model ({existing_model_name})")
                        self.logger.info(f"Performance improvement: {improvement_percentage:.4f}%")
                    
                    self.pipeline_results['model_registry'] = {
                        'status': 'REGISTERED',
                        'model_name': f"{model_name}_model",
                        'version': str(latest_version),
                        'alias': 'production',
                        'uri': model_uri,
                        'signature': str(signature),
                        'new_model_rmse': new_model_rmse,
                        'existing_model_name': existing_model_name if production_alias_exists else 'None',
                        'existing_model_rmse': existing_model_rmse if production_alias_exists else None,
                        'improvement_percentage': improvement_percentage if (production_alias_exists and existing_model_rmse) else None,
                        'replaced_previous': production_alias_exists
                    }
                    
                    return model_uri
                
                except Exception as e:
                    self.logger.error(f"✗ Error in Step 5 - Model Registration: {str(e)}")
                    raise
            
            # step 4: save pipeline
            with Timer('Save Pipeline', self.logger):
                output_dir = Path(self.config['file_paths'].get('metrics_artifacts', 'artifacts/models/metrics/'))
                output_dir.mkdir(parents=True, exist_ok=True)
                
                summary_path = output_dir / 'pipeline_summary.json'
                
                # Create comprehensive summary
                summary = {
                    'pipeline_metadata': {
                        'stage': 'model_training',
                        'objective': 'Train and select best model for production',
                        'validation_strategy': 'Cross-validation during hyperparameter tuning'
                    },
                    'data': self.pipeline_results.get('data_loading', {}),
                    'baseline_training': self.pipeline_results.get('baseline_training', {}),
                    'hyperparameter_tuning': self.pipeline_results.get('hyperparameter_tuning', {}),
                    'best_model': self.pipeline_results.get('best_model', {}),
                    'model_registry': self.pipeline_results.get('model_registry', {})
                }
                
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=4, default=str)
                
                mlflow.log_artifact(str(summary_path))
                
                self.logger.info(f"\n✓ Pipeline summary saved to {summary_path}")
                
                return str(summary_path)
            
            # step 5: save best model
            with Timer('Save Best Model', self.logger):
                output_dir = Path(self.config['file_paths'].get('metrics_artifacts', 'artifacts/models/metrics/'))
                output_dir.mkdir(parents=True, exist_ok=True)
                
                best_model_info_path = output_dir / 'best_model_info.json'
                
                best_model_info = {
                    'model_name': self.best_model_info['model_name'],
                    'hyperparameters': self.best_model_info['params'],
                    'train_metrics': {
                        'rmse': self.best_model_info['metrics']['rmse'],
                        'mse': self.best_model_info['metrics']['mse'],
                        'mae': self.best_model_info['metrics']['mae'],
                        'mape': self.best_model_info['metrics']['mape']
                    },
                    'tuning_info': {
                        'n_trials': self.best_model_info['tuning_info']['n_trials'],
                        'tuning_time_seconds': self.best_model_info['tuning_info']['tuning_time'],
                        'cv_best_rmse': self.best_model_info['tuning_info']['best_rmse']
                    },
                    'mlflow_registry': {
                        'registered_model_name': f"{self.best_model_info['model_name']}_model",
                        'alias': 'production',
                        'uri': f"models:/{self.best_model_info['model_name']}_model@production"
                    }
                }
                
                with open(best_model_info_path, 'w') as f:
                    json.dump(best_model_info, f, indent=4, default=str)
                
                mlflow.log_artifact(str(best_model_info_path))
                
                self.logger.info(f"✓ Best model info saved to {best_model_info_path}")
                
                return str(best_model_info_path)
            
            self.logger.info("\n" + "#"*70)
            self.logger.info("# MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("#"*70)
            self.logger.info(f"\nBest Model: {self.best_model_info['model_name']}")
            self.logger.info(f"Model Registry: {self.best_model_info['model_name']}_model")
            self.logger.info(f"Alias: production")
            self.logger.info(f"Train RMSE: {self.best_model_info['metrics']['rmse']:.4f}")
            self.logger.info(f"\nFiles Generated:")
            self.logger.info(f"  Summary: {summary_path}")
            self.logger.info(f"  Best Model Info: {best_model_info_path}")
            self.logger.info(f"\nTo load the production model in evaluation:")
            self.logger.info(f"  model = mlflow.pyfunc.load_model('{model_uri}')")

        except Exception as e:
            self.logger.error(f"\n✗ Pipeline execution failed: {str(e)}")
            mlflow.end_run(status='FAILED')
            raise

def main(config_path: str = 'config/training_config.yaml'):
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
    pipeline = ModelTrainingPipeline(config)
    pipeline.execute()
    



if __name__ == "__main__":
    # Run the pipeline
    main()
