import collections
import itertools
from pathlib import Path
from typing import Union, Optional
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from assignments.assignment1.b_data_profile import *
from assignments.assignment1.c_data_cleaning import *
from assignments.assignment1.d_data_encoding import *
from assignments.assignment1.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################


def process_iris_dataset() -> pd.DataFrame:
    """
    In this example, I call the methods you should have implemented in the other files
    to read and preprocess the iris dataset. This dataset is simple, and only has 4 columns:
    three numeric and one categorical. Depending on what I want to do in the future, I may want
    to transform these columns in other things (for example, I could transform a numeric column
    into a categorical one by splitting the number into bins, similar to how a histogram creates bins
    to be shown as a bar chart).

    In my case, what I want to do is to *remove missing numbers*, replacing them with valid ones,
    and *delete outliers* rows altogether (I could have decided to do something else, and this decision
    will be on you depending on what you'll do with the data afterwords, e.g. what machine learning
    algorithm you'll use). I will also standardize the numeric columns, create a new column with the average
    distance between the three numeric column and convert the categorical column to a onehot-encoding format.

    :return: A dataframe with no missing values, no outliers and onehotencoded categorical columns
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)

    for nc in numeric_columns:
        df = fix_outliers(df, nc)
        df = fix_nans(df, nc)
        df.loc[:, nc] = standardize_column(df.loc[:, nc])

    distances = pd.DataFrame()
    for nc_combination in list(itertools.combinations(numeric_columns, 2)):
        distances[str(nc_combination)] = calculate_numeric_distance(df.loc[:, nc_combination[0]],
                                                                    df.loc[:,
                                                                           nc_combination[1]],
                                                                    DistanceMetric.EUCLIDEAN).values
    df['numeric_mean'] = distances.mean(axis=1)

    for cc in categorical_columns:
        ohe = generate_one_hot_encoder(df.loc[:, cc])
        df = replace_with_one_hot_encoder(
            df, cc, ohe, list(ohe.get_feature_names()))

    return df


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def process_iris_dataset_again() -> pd.DataFrame:
    """
    Consider the example above and once again perform a preprocessing and cleaning of the iris dataset.
    This time, use normalization for the numeric columns and use label_encoder for the categorical column.
    Also, for this example, consider that all petal_widths should be between 0.0 and 1.0, replace the wong_values
    of that column with the mean of that column. Also include a new (binary) column called "large_sepal_lenght"
    saying whether the row's sepal_length is larger (true) or not (false) than 5.0
    :return: A dataframe with the above conditions.
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)

    for nc in numeric_columns:
        df = fix_outliers(df, nc)
        df = fix_nans(df, nc)
        df.loc[:, nc] = normalize_column(df.loc[:, nc])

    for cc in categorical_columns:
        le = generate_label_encoder(df.loc[:, cc])
        df = replace_with_label_encoder(df, cc, le)

    # Replacing wrong petal_widths with mean
    df = fix_numeric_wrong_values(
        df, "petal_width", WrongValueNumericRule.MUST_BE_GREATER_THAN, 0)
    df = fix_numeric_wrong_values(
        df, "petal_width", WrongValueNumericRule.MUST_BE_LESS_THAN, 1)
    df["petal_width"].fillna(df["petal_width"].mean(), inplace=True)

    # Adding large_sepal_length
    df.loc[(df["sepal_length"] > 5.0), "large_sepal_length"] = True
    df.loc[(df["sepal_length"] <= 5.0), "large_sepal_length"] = False

    return df


def fix_out_of_range_data_using_quantiles(df, column, range_min, range_max, min_quantile, max_quantile):
    df_copy = df.copy()

    quantile_min = df_copy[column].quantile(min_quantile)
    quantile_max = df_copy[column].quantile(max_quantile)

    df_copy.loc[(df_copy[column] < range_min), column] = quantile_min
    df_copy.loc[(df_copy[column] > range_max), column] = quantile_max

    return df_copy


def process_amazon_video_game_dataset():
    """
    Now use the rating_Video_Games dataset following these rules:
    1. The rating has to be between 1.0 and 5.0
    2. Time should be converted from milliseconds to datetime.datetime format
    3. For the future use of this data, I don't care about who voted what, I only want the average rating per product,
        therefore replace the user column by counting how many ratings each product had (which should be a column called count),
        and the average rating (as the "review" column).
    :return: A dataframe with the above conditions. The columns at the end should be: asin,review,time,count
    """
    df = read_dataset(Path('..', '..', 'ratings_Video_Games.csv'))

    # 1. Rating between 1.0 and 5.0. Using Flooring and Capping approach
    # Note: I am not using WrongValueNumericRule because they are not inclusive,
    # i.e., they'll leave out 1.0 and 5.0
    df = fix_out_of_range_data_using_quantiles(
        df, "review", 1.0, 5.0, 0.1, 0.9)

    # 2. Time to datetime
    df["time"] = pd.to_datetime(df["time"], unit="ms")

    # 3. Count and Average
    df.drop("user", axis=1, inplace=True)
    agg_df = df.groupby(["asin"]).agg(
        review=('review', np.mean), count=('asin', 'count'))
    # Merging time now because time is not required to be aggregated
    agg_df = agg_df.merge(df[['asin', 'time']], on="asin")

    processed_df = agg_df[['asin', 'review', 'time', 'count']]

    return processed_df


def process_amazon_video_game_dataset_again():
    """
    Now use the rating_Video_Games dataset following these rules (the third rule changed, and is more open-ended):
    1. The rating has to be between 1.0 and 5.0, drop any rows not following this rule
    2. Time should be converted from milliseconds to datetime.datetime format
    3. For the future use of this data, I just want to know more about the users, therefore show me how many reviews each user has,
        and a statistical analysis of each user (average, median, std, etc..., each as its own row)
    :return: A dataframe with the above conditions.
    """
    df = read_dataset(Path('..', '..', 'ratings_Video_Games.csv'))

    # 1. Rating between 1.0 and 5.0. Dropping other rows
    # Note: I am not using WrongValueNumericRule because they are not inclusive,
    # i.e., they'll leave out 1.0 and 5.0
    df.loc[((df["review"] > 5.0) | (df["review"] < 1.0)), "review"] = np.nan
    df = df[df['review'].notna()]

    # 2. Time to datetime
    df["time"] = pd.to_datetime(df["time"], unit="ms")

    # 3. Statistical Analysis
    agg_df = df.groupby(["user"]).agg(count=('review', 'count'), average=(
        'review', np.mean), median=('review', np.median), std=('review', np.std),
        min_rating=('review', np.min), max_rating=('review', np.max)).reset_index()
    agg_df["std"].fillna(0.0, inplace=True)

    return agg_df


def process_life_expectancy_dataset():
    """
    Now use the life_expectancy_years and geography datasets following these rules:
    1. The life expectancy dataset has missing values and outliers. Fix them.
    2. The geography dataset has problems with unicode letters. Make sure your code is handling it properly.
    3. Change the format of life expectancy, so that instead of one row with all 28 years, the data has 28 rows, one for each year,
        and with a column "year" with the year and a column "value" with the original value
    4. Merge (or more specifically, join) the two datasets with the common column being the country name (be careful with wrong values here)
    5. Drop all columns except country, continent, year, value and latitude (in this hypothetical example, we wish to analyse differences
        between southern and northern hemisphere)
    6. Change the latitude column from numerical to categorical (north vs south) and pass it though a label_encoder
    7. Change the continent column to a one_hot_encoder version of it
    :return: A dataframe with the above conditions.
    """
    df_geo = read_dataset(Path('..', '..', 'geography.csv'))
    df_ley = read_dataset(Path('..', '..', 'life_expectancy_years.csv'))
    all_columns_ley = list(df_ley.columns.values)

    # 1. fixing outlier and missing_values in ley dataset
    for column in all_columns_ley:
        df_ley = fix_outliers(df_ley, column)
        df_ley = fix_nans(df_ley, column)

    # 2. In Python 3, characters are unicode by default

    # 3. Change format of ley - from wide to long
    # Ref: https://www.journaldev.com/33398/pandas-melt-unmelt-pivot-function
    df_ley_rearranged = pd.melt(
        df_ley, id_vars=all_columns_ley[0], var_name="year", value_vars=all_columns_ley[1:])

    # 4. Merge ley and geo on country
    df_merged = df_ley_rearranged.merge(
        df_geo, left_on='country', right_on='name')

    # 5. Drop other columns except
    df_merged = df_merged[['country',
                           'eight_regions', 'year', 'value', 'Latitude']]
    # df_merged = pd.rename(df_merged, {"eight_regions"})

    # 6.


if __name__ == "__main__":
    # assert process_iris_dataset() is not None
    # assert process_iris_dataset_again() is not None
    # assert process_amazon_video_game_dataset() is not None
    # assert process_amazon_video_game_dataset_again() is not None
    assert process_life_expectancy_dataset() is not None
