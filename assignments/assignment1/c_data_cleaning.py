import collections
from pathlib import Path
from typing import Union, Optional
from enum import Enum

import pandas as pd
import numpy as np

from assignments.assignment1.b_data_profile import *
from assignments.assignment1.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
class WrongValueNumericRule(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    MUST_BE_POSITIVE = 0
    MUST_BE_NEGATIVE = 1
    MUST_BE_GREATER_THAN = 2
    MUST_BE_LESS_THAN = 3


class DistanceMetric(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    EUCLIDEAN = 0
    MANHATTAN = 1


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def fix_numeric_wrong_values(df: pd.DataFrame,
                             column: str,
                             must_be_rule: WrongValueNumericRule,
                             must_be_rule_optional_parameter: Optional[float] = None) -> pd.DataFrame:
    """
    This method should fix the wrong_values depending on the logic you think best to do and using the rule passed by parameter.
    Remember that wrong values are values that are in the dataset, but are wrongly inputted (for example, a negative age).
    Here you should only fix them (and not find them) by replacing them with np.nan ("not a number", or also called "missing value")
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :param must_be_rule: one of WrongValueNumericRule identifying what rule should be followed to flag a value as a wrong value
    :param must_be_rule_optional_parameter: optional parameter for the "greater than" or "less than" cases
    :return: The dataset with fixed column
    """
    df_copy = df.copy()

    if must_be_rule == WrongValueNumericRule.MUST_BE_LESS_THAN:
        df_copy.loc[df_copy[column] >=
                    must_be_rule_optional_parameter, column] = np.nan
    elif must_be_rule == WrongValueNumericRule.MUST_BE_GREATER_THAN:
        df_copy.loc[df_copy[column] <=
                    must_be_rule_optional_parameter, column] = np.nan
    elif must_be_rule == WrongValueNumericRule.MUST_BE_NEGATIVE:
        df_copy.loc[df_copy[column] >= 0, column] = np.nan
    elif must_be_rule == WrongValueNumericRule.MUST_BE_POSITIVE:
        df_copy.loc[df_copy[column] < 0, column] = np.nan

    return df_copy


def fix_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This method should fix the column in respective to outliers depending on the logic you think best to do.
    Feel free to choose which logic you prefer, but if you are in doubt, use the simplest one to remove the row
    of the dataframe which is an outlier (note that the issue with this approach is when the dataset is small,
    dropping rows will make it even smaller).
    Remember that some datasets are large, and some are small, so think wisely on how to calculate outliers
    and when to remove/replace them. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The dataset with fixed column
    """
    numeric_columns = get_numeric_columns(df)
    if column in numeric_columns:
        # Ref: https://app.pluralsight.com/guides/cleaning-up-data-from-outliers
        # Finding outliers
        quartile1 = df[column].quantile(0.25)
        quartile3 = df[column].quantile(0.75)
        inter_quartile_range = quartile3 - quartile1

        df["Is_Outlier"] = (df[column] < (quartile1 - 1.5 * inter_quartile_range)
                            ) | (df[column] > (quartile3 + 1.5 * inter_quartile_range))

        # Fixing outliers - Flooring and Capping approach
        quantile10 = df[column].quantile(0.10)
        quantile90 = df[column].quantile(0.90)
        df.loc[((df["Is_Outlier"] == True) & (
            df[column] < quantile10)), column] = quantile10
        df.loc[((df["Is_Outlier"] == True) & (
            df[column] > quantile90)), column] = quantile90
        df.drop("Is_Outlier", axis=1, inplace=True)

        return df

    print("Not a numeric column")
    return df


def fix_nan_by_column_type(df: pd.DataFrame, column: str) -> pd.DataFrame:
    bin_cols = get_binary_columns(df)
    num_cols = get_numeric_columns(df)
    text_cols = df.select_dtypes(include='object')
    datetime_cols = df.select_dtypes(include='datetime')

    if any(column in cols for cols in [bin_cols, text_cols, datetime_cols]):
        # Drop rows if column type is either of binary, text_categorical, or datetime
        df = df[df[column].notna()]

    elif column in num_cols:
        # Fill nan with median values
        df[column].fillna(df[column].median(), inplace=True)

    return df


def fix_nans(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This method should fix all nans (missing data) depending on the logic you think best to do
    Remember that some datasets are large, and some are small, so think wisely on when to use each possible
    removal/fix/replace of nans. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The fixed dataset
    """
    missing_values_percentage = (df[column].isna().sum() / len(df)) * 100
    # Delete column if percentage of missing values is at least 50
    if missing_values_percentage >= 50:
        df.drop(column, axis=1, inplace=True)
    else:
        df = fix_nan_by_column_type(df, column)

    return df


def normalize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and normalise it between 0 and 1.
    :param df_column: Dataset's column
    :return: The column normalized
    """
    min_value = df_column.min()
    max_value = df_column.max()
    norm_data = (df_column - min_value) / (max_value - min_value)

    return norm_data


def standardize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and standardize it between -1 and 1 with its average at 0.
    :param df_column: Dataset's column
    :return: The column standardized
    """
    mean_value = df_column.mean()
    std_dev = df_column.std()
    std_data = (df_column - mean_value) / std_dev

    return std_data


def calculate_numeric_distance(df_column_1: pd.Series, df_column_2: pd.Series, distance_metric: DistanceMetric) -> pd.Series:
    """
    This method should calculate the distance between two numeric columns
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :param distance_metric: One of DistanceMetric, and for each one you should implement its logic
    :return: A new 'column' with the distance between the two inputted columns
    """
    if distance_metric == DistanceMetric.EUCLIDEAN:
        distance = np.linalg.norm(df_column_1 - df_column_2)

    elif distance_metric == DistanceMetric.MANHATTAN:
        # TODO

    else:
        print("Distance metric not supported")
        distance = None

    pass


def calculate_binary_distance(df_column_1: pd.Series, df_column_2: pd.Series) -> pd.Series:
    """
    This method should calculate the distance between two binary columns.
    Choose one of the possibilities shown in class for future experimentation.
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :return: A new 'column' with the distance between the two inputted columns
    """
    pass


if __name__ == "__main__":
    df = pd.DataFrame({'a': [1, 2, 3, None], 'b': [
        True, True, False, None], 'c': [1, 2, np.nan, 4]})
    # df = pd.DataFrame({'a': [1, 2, 3, None], 'b': [
    #                   True, True, False, None], 'c': ['one', 'two', np.nan, None]})
    assert fix_numeric_wrong_values(
        df, 'a', WrongValueNumericRule.MUST_BE_LESS_THAN, 2) is not None
    assert fix_numeric_wrong_values(
        df, 'a', WrongValueNumericRule.MUST_BE_GREATER_THAN, 2) is not None
    assert fix_numeric_wrong_values(
        df, 'a', WrongValueNumericRule.MUST_BE_POSITIVE, 2) is not None
    assert fix_numeric_wrong_values(
        df, 'a', WrongValueNumericRule.MUST_BE_NEGATIVE, 2) is not None
    assert fix_outliers(df, 'c') is not None
    assert fix_nans(df, 'c') is not None
    assert normalize_column(df.loc[:, 'a']) is not None
    assert standardize_column(df.loc[:, 'a']) is not None
    assert calculate_numeric_distance(
        df.loc[:, 'a'], df.loc[:, 'a'], DistanceMetric.EUCLIDEAN) is not None
    assert calculate_numeric_distance(
        df.loc[:, 'a'], df.loc[:, 'a'], DistanceMetric.MANHATTAN) is not None
    assert calculate_binary_distance(
        df.loc[:, 'b'], df.loc[:, 'b']) is not None
    print("ok")
