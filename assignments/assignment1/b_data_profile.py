from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from assignments.assignment1.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
def pandas_profile(df: pd.DataFrame, result_html: str = 'report.html'):
    """
    This method will be responsible to extract a pandas profiling report from the dataset.
    Do not change this method, but run it and look through the html report it generated.
    Always be sure to investigate the profile of your dataset (max, min, missing values, number of 0, etc).
    """
    from pandas_profiling import ProfileReport

    profile = ProfileReport(df, title="Pandas Profiling Report")
    if result_html is not None:
        profile.to_file(result_html)
    return profile.to_json()


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def get_column_max(df: pd.DataFrame, column_name: str) -> float:
    pass


def get_column_min(df: pd.DataFrame, column_name: str) -> float:
    pass


def get_column_mean(df: pd.DataFrame, column_name: str) -> float:
    pass


def get_column_count_of_nan(df: pd.DataFrame, column_name: str) -> float:
    """
    This is also known as the number of 'missing values'
    """
    pass


def get_column_number_of_duplicates(df: pd.DataFrame, column_name: str) -> float:
    pass


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    pass


def get_binary_columns(df: pd.DataFrame) -> List[str]:
    pass


def get_text_categorical_columns(df: pd.DataFrame) -> List[str]:
    pass


def get_correlation_between_columns(df: pd.DataFrame, col1: str, col2: str) -> float:
    """
    Calculate and return the pearson correlation between two columns
    """
    pass


if __name__ == "__main__":
    df = read_dataset(Path('..', '..', 'iris.csv'))
    a = pandas_profile(df)
    assert get_column_max(df, df.columns[0]) is not None
    assert get_column_min(df, df.columns[0]) is not None
    assert get_column_mean(df, df.columns[0]) is not None
    assert get_column_count_of_nan(df, df.columns[0]) is not None
    assert get_column_number_of_duplicates(df, df.columns[0]) is not None
    assert get_numeric_columns(df) is not None
    assert get_binary_columns(df) is not None
    assert get_text_categorical_columns(df) is not None
    assert get_correlation_between_columns(df, df.columns[0], df.columns[1]) is not None
