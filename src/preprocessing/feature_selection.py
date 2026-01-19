"""Feature selection utilities for preprocessing.
Workflow:
1. Import necessary libraries.
2. Use scikit-learn "SelectKBest" for feature selection.
3. PCA for dimensionality reduction.
4. Define fit and transform functions.
5. Save fitted selector as a joblib file
6. Log artifacts using mlflow."""