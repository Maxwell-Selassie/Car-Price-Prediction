"""  
Module for splitting data into training, development and testing sets.

Workflow:
1. Import necessary libraries.
2. Define a function to split the data
into train, dev and test sets.
3. Return the split datasets for further preprocessing
Test size = 1000 samples
Dev size = 1000 samples
Train size = Remaining samples
4. Save the split datasets as CSV files for future use.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from utils import LoggerMixin
from pathlib import Path