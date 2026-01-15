"""Visualization functions for exploratory data analysis.

Workflows:
1. Plot distribution of numeric columns
2. Plot count plots for categorical columns
3. Plot correlation heatmap for numeric columns
4. Plot box plots for numeric columns
5. Plot scatter plots for pair of numeric columns
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict
from utils import LoggerMixin, ensure_directory
import mlflow

class EDAVisualizer(LoggerMixin):

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = self.setup_class_logger("EDAVisualizer", config, "logging")
        self.ouput_dir = Path(config['file_paths'].get("figures_dir", "eda_plots"))
        ensure_directory(self.ouput_dir)
        

    def plot_numeric_distribution(self, df: pd.DataFrame) -> None:
        """  
        Plot distribution of numeric columns in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
        """
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            self.logger.warning("No numeric columns found.")
            return
        

        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes] 

        for idx, col in enumerate(numeric_cols):
            try:
                sns.histplot(
                    data=df,
                    x=col,
                    kde=True,
                    ax=axes[idx],
                    color='purple',
                    alpha=0.6,
                )
                axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
            except Exception as e:
                self.logger.error(f'Error plotting {col}: {e}')
                axes[idx].text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                            ha='center', va='center', transform=axes[idx].transAxes)
        
        # Hide empty subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if self.save_plots:
            output_file = self.output_dir / 'numeric_distributions.png'
            dpi = self.config['output'].get('plot_dpi', 300)
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            mlflow.log_artifact(output_file)
            plt.close(fig)
            self.logger.info(f'✓ Saved: {output_file}')
        else:
            plt.show()

    def plot_categorical_counts(self, df: pd.DataFrame) -> None:
        """  
        Plot count plots for categorical columns in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
        """
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        if not categorical_cols:
            self.logger.warning("No categorical columns found.")
            return

        n_cols = len(categorical_cols)
        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes] 

        for idx, col in enumerate(categorical_cols):
            try:
                sns.countplot(
                    data=df,
                    x=col,
                    ax=axes[idx],
                    palette='Set2',
                )
                axes[idx].set_title(f'Count Plot of {col}', fontweight='bold')
                axes[idx].set_ylabel('Count')
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].grid(True, alpha=0.3)
            except Exception as e:
                self.logger.error(f'Error plotting {col}: {e}')
                axes[idx].text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                            ha='center', va='center', transform=axes[idx].transAxes)
        
        # Hide empty subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if self.save_plots:
            output_file = self.output_dir / 'categorical_counts.png'
            dpi = self.config['output'].get('plot_dpi', 300)
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            mlflow.log_artifact(output_file)
            plt.close(fig)
            self.logger.info(f'✓ Saved: {output_file}')
        else:
            plt.show()

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """  
        Plot correlation heatmap for numeric columns in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
        """
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            self.logger.warning("Not enough numeric columns for correlation heatmap.")
            return

        corr_matrix = df[numeric_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            square=True,
            cbar_kws={"shrink": .8},
            annot_kws={"size": 8}
        )
        plt.title('Correlation Heatmap', fontweight='bold')
        plt.tight_layout()
        
        if self.save_plots:
            output_file = self.output_dir / 'correlation_heatmap.png'
            dpi = self.config['output'].get('plot_dpi', 300)
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            mlflow.log_artifact(output_file)
            plt.close()
            self.logger.info(f'✓ Saved: {output_file}')
        else:
            plt.show()

    def plot_box_plots(self, df: pd.DataFrame) -> None: 
        """  
        Plot box plots for numeric columns in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
        """
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            self.logger.warning("No numeric columns found.")
            return

        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes] 

        for idx, col in enumerate(numeric_cols):
            try:
                sns.boxplot(
                    data=df,
                    x=col,
                    ax=axes[idx],
                    color='lightblue',
                )
                axes[idx].set_title(f'Box Plot of {col}', fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
            except Exception as e:
                self.logger.error(f'Error plotting {col}: {e}')
                axes[idx].text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                            ha='center', va='center', transform=axes[idx].transAxes)
        
        # Hide empty subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if self.save_plots:
            output_file = self.output_dir / 'box_plots.png'
            dpi = self.config['output'].get('plot_dpi', 300)
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            mlflow.log_artifact(output_file)
            plt.close(fig)
            self.logger.info(f'✓ Saved: {output_file}')
        else:
            plt.show()

    def plot_scatter_plots(self, df: pd.DataFrame, pairs: List[Tuple[str, str]]) -> None:
        """  
        Plot scatter plots for given pairs of numeric columns in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
            pairs (List[Tuple[str, str]]): List of tuples containing pairs of column names.
        """
        n_pairs = len(pairs)
        n_rows = (n_pairs + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_pairs > 1 else [axes] 

        for idx, (x_col, y_col) in enumerate(pairs):
            try:
                sns.scatterplot(
                    data=df,
                    x=x_col,
                    y=y_col,
                    ax=axes[idx],
                    color='green',
                    alpha=0.6,
                )
                axes[idx].set_title(f'Scatter Plot of {y_col} vs {x_col}', fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
            except Exception as e:
                self.logger.error(f'Error plotting {x_col} vs {y_col}: {e}')
                axes[idx].text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                            ha='center', va='center', transform=axes[idx].transAxes)
        
        # Hide empty subplots
        for idx in range(n_pairs, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if self.save_plots:
            output_file = self.output_dir / 'scatter_plots.png'
            dpi = self.config['output'].get('plot_dpi', 300)
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            mlflow.log_artifact(output_file)
            plt.close(fig)
            self.logger.info(f'✓ Saved: {output_file}')
        else:
            plt.show()

    def run_all_visualizations(self, df: pd.DataFrame, scatter_pairs: Optional[List[Tuple[str, str]]] = None) -> None:
        """  
        Run all visualization methods.

        Args:
            df (pd.DataFrame): Input dataframe.
            scatter_pairs (Optional[List[Tuple[str, str]]]): List of tuples for scatter plot pairs.
        """
        self.plot_numeric_distribution(df)
        self.plot_categorical_counts(df)
        self.plot_correlation_heatmap(df)
        self.plot_box_plots(df)
        if scatter_pairs:
            self.plot_scatter_plots(df, scatter_pairs)
    
    