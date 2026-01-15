"""   
Descriptive statistics functions for exploratory data analysis (EDA).
workflows:
1. Summary Statistics for numeric column
2. Summary Statistics for categorical column
"""

import pandas as pd
import numpy as np
from utils import LoggerMixin
import mlflow
from typing import Dict, List, Any
from pathlib import Path

class DescriptiveStats(LoggerMixin):
    """Class for computing descriptive statistics for EDA."""

    def __init__(self, config: Dict[str, Any] ) -> None:
        super().__init__()
        self.config = config
        self.logger = self.setup_class_logger('DescriptiveStats', config, 'logging')

    def summary_numeric(self, df: pd.DataFrame, output_dir: str | Path) -> List[str]:
        """Compute summary statistics for a numeric column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            output_dir (str | Path): Directory to save the output.

        Returns:
            List of numeric columns
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            self.logger.warning("No numeric columns found.")
            return []
        
        summary_stats = df[numeric_cols].describe().T
        summary_stats['variance'] = df[numeric_cols].var()
        summary_stats['skewness'] = df[numeric_cols].skew()
        summary_stats['kurtosis'] = df[numeric_cols].kurtosis()
        try:
            output_path = Path(output_dir) / "numeric_summary_statistics.csv"
            summary_stats.to_json(output_path)
            self.logger.info(f"Numeric summary statistics saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving numeric summary statistics: {e}")
            raise e

        mlflow.log_artifact(str(output_path))
        mlflow.set_tag("mlflow.note.content",
                f"Numeric summary statistics computed for {len(numeric_cols)} columns. - See artifact: {output_path}")
        return numeric_cols
    
    def summary_categorical(self, df: pd.DataFrame, output_dir: str | Path) -> List[str]:
        """Compute summary statistics for a categorical column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            output_dir (str | Path): Directory to save the output.
        Returns:
            List of categorical columns
        """
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not categorical_cols:
            self.logger.warning("No categorical columns found.")
            return []

        try:
            output_path = Path(output_dir) / "categorical_summary_statistics.json"
            df.describe(exclude=[np.number]).T.to_json(output_path)
            self.logger.info(f"Categorical summary statistics saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving categorical summary statistics: {e}")
            raise e

        mlflow.log_artifact(str(output_path))
        mlflow.note.content("mlflow.note.content",
                f"Categorical summary statistics computed for {len(categorical_cols)} columns. - See artifact: {output_path}")
        return categorical_cols
    
