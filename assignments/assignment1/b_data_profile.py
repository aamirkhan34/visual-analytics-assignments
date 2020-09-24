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
    unwanted_dtypes = [str, object, pd.datetime]

    if df[column_name].dtype not in unwanted_dtypes:
        return df[column_name].max()

    raise ValueError("Datatype is not numeric")


def get_column_min(df: pd.DataFrame, column_name: str) -> float:
    unwanted_dtypes = [str, object, pd.datetime]

    if df[column_name].dtype not in unwanted_dtypes:
        return df[column_name].min()

    raise ValueError("Datatype is not numeric")


def get_column_mean(df: pd.DataFrame, column_name: str) -> float:
    unwanted_dtypes = [str, object, pd.datetime]

    if df[column_name].dtype not in unwanted_dtypes:
        return df[column_name].mean()

    raise ValueError("Datatype is not numeric")


def get_column_count_of_nan(df: pd.DataFrame, column_name: str) -> float:
    """
    This is also known as the number of 'missing values'
    """
    return df[column_name].isna().sum()


def get_column_number_of_duplicates(df: pd.DataFrame, column_name: str) -> float:
    all_values = df[column_name].tolist()
    unique_values = set(all_values)

    duplicates_count = len(all_values) - len(unique_values)

    return duplicates_count


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    df_numeric = df.select_dtypes(include='number')

    return list(df_numeric.columns.values)


def get_binary_columns(df: pd.DataFrame) -> List[str]:
    df_binary = df.select_dtypes(include='bool')

    return list(df_binary.columns.values)


def get_text_categorical_columns(df: pd.DataFrame) -> List[str]:
    df_text = df.select_dtypes(include='object')
    df_text_cat = df.select_dtypes(include='category')

    return list(df_text_cat.columns.values)


def get_correlation_between_columns(df: pd.DataFrame, col1: str, col2: str) -> float:
    """
    Calculate and return the pearson correlation between two columns
    """
    correlation_score = df[col1].corr(df[col2], method='pearson')

    return correlation_score


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
    assert get_correlation_between_columns(
        df, df.columns[0], df.columns[1]) is not None
