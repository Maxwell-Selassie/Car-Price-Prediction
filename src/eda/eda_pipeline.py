""" EDA Pipeline Module
This module defines an EDA pipeline class that orchestrates various exploratory data analysis
techniques.
It visualization methods, and data summarization techniques into a cohesive workflow.
"""

import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from typing import List, Optional
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import ensure_directory, read_yaml, Timer, setup_logger
from eda import (
    DataLoader, DataQualityChecker, EDAVisualizer, DescriptiveStats
)

class EDAPipeline:
    """Exploratory Data Analysis Pipeline"""
    def __init__(self, config_path: str | Path):
        super().__init__()
        self.config_path = config_path
        self.logger = setup_logger(
            name='EDAPipeline',
            log_dir=Path('logs/')
        )

        self.results = {}

        self.logger.info('='*80)
        self.logger.info('EDA PIPELINE INITIALIZED')    
        self.logger.info('='*80)


    def execute(self) -> dict:
        """Execute the EDA pipeline"""
        try:
            with mlflow.start_run(run_name="EDA_Pipeline") as run:
                mlflow.set_tag("stage", "EDA")

                # Load configuration
                config = read_yaml(self.config_path)
                data_path = config["file_paths"].get('raw_data', 'data/raw/car_details.csv')
                
                # Data Loading
                with mlflow.start_run(run_name="Data_Loading", nested=True):
                    with Timer('Data Loading', self.logger):
                        data_loader = DataLoader(config)
                        df = data_loader.load_dataset(data_path)
                        self.logger.info(f"Data loaded with shape: {df.shape}")

                # Data Quality Checks
                with mlflow.start_run(run_name="Data_Quality_Checks", nested=True):
                    with Timer('Data Quality Checks', self.logger):
                        dq_checker = DataQualityChecker(config)
                        quality_report = dq_checker.run_all_checks(df, 
                            expected_columns=config['data_quality_checks']["expected_columns"],
                            target_column=config['data_quality_checks']["target_column"],
                            output_dir=Path(config["file_paths"].get("eda_reports", "data_quality_reports"))
                        )
                        dq_checker.save_validation_report()
                        self.results['data_quality'] = quality_report
                        self.logger.info("Data quality checks completed")

                # Descriptive Statistics
                with mlflow.start_run(run_name="Descriptive_Statistics", nested=True):
                    with Timer('Descriptive Statistics', self.logger):
                        desc_stats = DescriptiveStats(config)
                        stats_report_numeric = desc_stats.summary_numeric()
                        stats_report_categorical = desc_stats.summary_categorical()
                        self.results['descriptive_statistics'] = {
                            'numeric_stats_reports' : stats_report_numeric,
                            'categorical_stats_reports' : 
                        }
                        self.logger.info("Descriptive statistics generated")

                # Visualizations
                with mlflow.start_run(run_name="Visualizations", nested=True):
                    with Timer('Visualizations', self.logger)
                        visualizer = EDAVisualizer(df, 
                        output_dir=Path(config['visualizations']['output_dir']))
                        visualizer.generate_all_plots()
                        self.logger.info("Visualizations generated")

                self.logger.info("EDA Pipeline Execution Completed Successfully")
                
            return self.results

        except Exception as e:
            self.logger.error(f"EDA Pipeline Execution Failed: {e}")
            raise
    
    