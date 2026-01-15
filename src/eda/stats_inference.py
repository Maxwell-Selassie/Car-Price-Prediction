# """
# Statistical Inference Functions for EDA

# This module provides functions to perform statistical inference tests
# commonly used in exploratory data analysis (EDA). It includes functions for
# t-tests, chi-squared tests, ANOVA, and correlation analysis.

# Workflows:
# 1. t-test: Compare means between two groups.
# 2. Chi-squared test: Test independence between categorical variables.
# 3. ANOVA: Compare means across multiple groups.
# 4. Correlation analysis: Measure the strength and direction of relationships
# between continuous variables.
# 5. confidence intervals: Calculate confidence intervals for means and proportions.
# 6. effect size: Compute effect sizes for various statistical tests.
# """

# import numpy as np
# import pandas as pd
# import scipy.stats as stats
# from typing import Tuple, List, Union
# import mlflow 
# from pathlib import Path
# from utils import LoggerMixin, ensure_directory
# import json

# class StatsInference(LoggerMixin):
#     def __init__(self, experiment_name: str = "StatsInference"):
#         super().__init__()
#         self.experiment_name = experiment_name
#         mlflow.set_experiment(experiment_name)

#     def t_test(self, group1: np.ndarray, group2: np.ndarray, equal_var: bool = True) -> Tuple[float, float]:
#         """
#         Perform a t-test to compare the means of two independent groups.

#         Parameters:
#         group1 (np.ndarray): Data for the first group.
#         group2 (np.ndarray): Data for the second group.
#         equal_var (bool): If True, perform a standard t-test; if False, perform Welch's t-test.

#         Returns:
#         Tuple[float, float]: t-statistic and p-value.
#         """
#         t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
#         self.logger.info(f"T-test results: t-statistic={t_stat}, p-value={p_value}")
#         return t_stat, p_value

#     def chi_squared_test(self, observed: np.ndarray) -> Tuple[float, float]:
#         """
#         Perform a chi-squared test for independence.

#         Parameters:
#         observed (np.ndarray): Contingency table of observed frequencies.

#         Returns:
#         Tuple[float, float]: chi-squared statistic and p-value.
#         """
#         chi2_stat, p_value, _, _ = stats.chi2_contingency(observed)
#         self.logger.info(f"Chi-squared test results: chi2-statistic={chi2_stat}, p-value={p_value}")
#         return chi2_stat, p_value

#     def anova(self, *groups: np.ndarray) -> Tuple[float, float]:
#         """
#         Perform one-way ANOVA to compare means across multiple groups.

#         Parameters:
#         *groups (np.ndarray): Data for each group.

#         Returns:
#         Tuple[float, float]: F-statistic and p-value.
#         """
#         f_stat, p_value = stats.f_oneway(*groups)
#         self.logger.info(f"ANOVA results: F-statistic={f_stat}, p-value={p_value}")
#         return f_stat, p_value

#     def correlation_analysis(self, x: np.ndarray, y: np.ndarray, method: str = 'pearson') -> Tuple[float, float]:
#         """
#         Perform correlation analysis between two continuous variables.

#         Parameters:
#         x (np.ndarray): First variable.
#         y (np.ndarray): Second variable.    
#         method (str): Correlation method ('pearson', 'spearman', 'kendall').
#         Returns:
#         Tuple[float, float]: correlation coefficient and p-value.
#         """
#         if method == 'pearson':
#             corr_coef, p_value = stats.pearsonr(x, y)
#         elif method == 'spearman':
#             corr_coef, p_value = stats.spearmanr(x, y)
#         elif method == 'kendall':
#             corr_coef, p_value = stats.kendalltau(x, y)
#         else:
#             raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'.")
        
#         self.logger.info(f"Correlation analysis results ({method}): correlation coefficient={corr_coef}, p-value={p_value}")
#         return corr_coef, p_value
    
#     def confidence_interval(self, data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
#         """
#         Calculate the confidence interval for the mean of the data.

#         Parameters:
#         data (np.ndarray): Data for which to calculate the confidence interval.
#         confidence (float): Confidence level.

#         Returns:
#         Tuple[float, float]: Lower and upper bounds of the confidence interval.
#         """
#         n = len(data)
#         mean = np.mean(data)
#         sem = stats.sem(data)
#         h = sem * stats.t.ppf((1 + confidence) / 2., n-1)
#         ci_lower = mean - h
#         ci_upper = mean + h
#         self.logger.info(f"Confidence interval ({confidence*100}%): [{ci_lower}, {ci_upper}]")
#         return ci_lower, ci_upper
    
#     def effect_size(self, group1: np.ndarray, group2: np.ndarray) -> float:
#         """
#         Compute Cohen's d effect size between two groups.

#         Parameters:
#         group1 (np.ndarray): Data for the first group.
#         group2 (np.ndarray): Data for the second group.

#         Returns:
#         float: Cohen's d effect size.
#         """
#         mean1 = np.mean(group1)
#         mean2 = np.mean(group2)
#         pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1) + (len(group2) - 1) * np.var(group2)) / (len(group1) + len(group2) - 2))
#         cohen_d = (mean1 - mean2) / pooled_std
#         self.logger.info(f"Cohen's d effect size: {cohen_d}")
#         return cohen_d
    
#     def save_results(self, results: dict, filename: str) -> None:
#         """Save the statistical results to a JSON file."""
#         ensure_directory(Path(filename).parent)
#         with open(filename, 'w') as f:
#             json.dump(results, f, indent=4)
#         self.logger.info(f"Results saved to {filename}")
    
    