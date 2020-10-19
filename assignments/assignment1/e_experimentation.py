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
def move_target_col_to_end(x_df, target_col) -> pd.DataFrame:
    # Put target column (species) at the end
    cols = list(x_df.columns.values)
    cols.remove(target_col)
    x_df = x_df[cols+[target_col]]

    return x_df


def prepare_iris_for_ml_model(iris_df, model_type, categorical_columns) -> pd.DataFrame:
    if model_type == "classification":
        for cc in categorical_columns:
            le = generate_label_encoder(iris_df.loc[:, cc])
            iris_df = replace_with_label_encoder(iris_df, cc, le)
        iris_df = move_target_col_to_end(iris_df, 'species')

    elif model_type == "regression":
        for cc in categorical_columns:
            ohe = generate_one_hot_encoder(iris_df.loc[:, cc])
            iris_df = replace_with_one_hot_encoder(
                iris_df, cc, ohe, list(ohe.get_feature_names()))
        iris_df = move_target_col_to_end(iris_df, 'sepal_length')

    elif model_type == "clustering":
        for cc in categorical_columns:
            ohe = generate_one_hot_encoder(iris_df.loc[:, cc])
            iris_df = replace_with_one_hot_encoder(
                iris_df, cc, ohe, list(ohe.get_feature_names()))

    return iris_df


def process_iris_dataset_again(model_type: str) -> pd.DataFrame:
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

    # Replacing wrong petal_widths with mean
    # Cannot use WrongValueNumericRule because it doesn't include the extreme values
    df.loc[((df["petal_width"] < 0.0) | (df["petal_width"] > 1.0)),
           "petal_width"] = np.nan
    df["petal_width"].fillna(df["petal_width"].mean(), inplace=True)

    # Adding large_sepal_length
    # Cannot use WrongValueNumericRule because it doesn't include the extreme values
    df.loc[(df["sepal_length"] > 5.0), "large_sepal_length"] = 1
    df.loc[(df["sepal_length"] <= 5.0), "large_sepal_length"] = 0

    for nc in numeric_columns:
        df = fix_outliers(df, nc)
        df = fix_nans(df, nc)
        if nc == 'sepal_length' and model_type == 'regression':
            continue
        df.loc[:, nc] = normalize_column(df.loc[:, nc])

    df = prepare_iris_for_ml_model(df, model_type, categorical_columns)

    return df


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

    processed_df = agg_df[['asin', 'review', 'count']]

    le = generate_label_encoder(processed_df.loc[:, "asin"])
    processed_df = replace_with_label_encoder(processed_df, "asin", le)

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
        'review', np.mean), median=('review', np.median), std=('review', np.std)).reset_index()
    agg_df["std"].fillna(0.0, inplace=True)

    agg_df = agg_df.merge(df[['user', 'review']], on="user")

    # Put target column (user) at the end
    cols = list(agg_df.columns.values)
    cols.remove('user')
    agg_df = agg_df[cols+['user']]

    return agg_df


def prepare_life_expectancy_data_for_ml_model(le_df, model_type) -> pd.DataFrame:
    if model_type == 'classification':
        le_df.drop(['year'], inplace=True, axis=1)
        le_df = move_target_col_to_end(le_df, 'latitude')

    elif model_type == 'regression':
        le_df = move_target_col_to_end(le_df, 'value')

    elif model_type == 'clustering':
        le_df.drop(['year'], inplace=True, axis=1)
        le_df = le_df.drop_duplicates()
        for c in list(le_df.columns):
            le_df[c] = normalize_column(le_df[c])

    return le_df


def process_life_expectancy_dataset(model_type: str):
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
                           'four_regions', 'year', 'value', 'Latitude']]
    df_merged = df_merged.rename(
        columns={"four_regions": "continent", "Latitude": "latitude"})

    # 6. latitude: numerical to categorical (north vs south)
    # Ref: https://stackoverflow.com/questions/34630916/which-hemisphere-based-on-latitude-and-longitude
    df_merged.loc[(df_merged["latitude"] >= 0.0), "latitude_cat"] = "north"
    df_merged.loc[(df_merged["latitude"] < 0.0), "latitude_cat"] = "south"
    df_merged.drop("latitude", axis=1, inplace=True)
    df_merged = df_merged.rename(columns={"latitude_cat": "latitude"})
    le = generate_label_encoder(df_merged["latitude"])
    df_merged = replace_with_label_encoder(df_merged, "latitude", le)

    # 7. continent column to a one_hot_encoder version of it
    ohe = generate_one_hot_encoder(df_merged["continent"])
    ohe_col_names = get_one_hot_encoding_column_names(
        df_merged["continent"], "continent")
    df_merged = replace_with_one_hot_encoder(
        df_merged, "continent", ohe, ohe_col_names)

    # OHE country
    ohe2 = generate_one_hot_encoder(df_merged["country"])
    df_merged = replace_with_one_hot_encoder(
        df_merged, "country", ohe2, list(ohe2.get_feature_names()))

    df_merged = prepare_life_expectancy_data_for_ml_model(
        df_merged, model_type)

    return df_merged


if __name__ == "__main__":
    assert process_iris_dataset() is not None
    assert process_iris_dataset_again() is not None
    assert process_amazon_video_game_dataset() is not None
    assert process_amazon_video_game_dataset_again() is not None
    assert process_life_expectancy_dataset() is not None
    print("ok")
