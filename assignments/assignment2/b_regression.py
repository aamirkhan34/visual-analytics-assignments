from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.d_data_encoding import fix_outliers, fix_nans, normalize_column, \
    generate_one_hot_encoder, replace_with_one_hot_encoder
from assignments.assignment1.e_experimentation import process_iris_dataset_again, process_amazon_video_game_dataset, process_life_expectancy_dataset, move_target_col_to_end

"""
Regression is a supervised form of machine learning. It uses labeled data, which is data with an expected
result available, and uses it to train a machine learning model to predict the said result. Regression
focuses in results of the numerical type.
"""


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
def simple_random_forest_regressor(X: pd.DataFrame, y: pd.Series) -> Dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # If necessary, change the n_estimators, max_depth and max_leaf_nodes in the below method to accelerate the model training,
    # but don't forget to comment why you did and any consequences of setting them!
    # Now I am doing a regression!
    # Reduced the number of estimators to 10 for faster and more efficient fitting.
    # It will load only 10 trees in ensemble.
    model = RandomForestRegressor(n_estimators=10)
    model.fit(X_train, y_train)
    # Use this line to get the prediction from the model
    y_predict = model.predict(X_test)

    # In regression, there is no accuracy, but other types of score. See the following link for one example (R^2)
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor.score
    score = model.score(X_test, y_test)

    return dict(model=model, score=score, test_prediction=y_predict)


def simple_random_forest_on_iris() -> Dict:
    """
    Here I will run a regression on the iris dataset with a random-forest regressor
    Notice that my logic has changed. I am not predicting the species anymore, but
    am predicting the sepal_length. I am also removing the species column, and will handle
    it in the next example.
    """
    df = pd.read_csv(Path('..', '..', 'iris.csv'))
    X, y = df.iloc[:, 1:4], df.iloc[:, 0]
    return simple_random_forest_regressor(X, y)


def reusing_code_random_forest_on_iris() -> Dict:
    """
    Again I will run a regression on the iris dataset, but reusing
    the existing code from assignment1. I am also including the species column as a one_hot_encoded
    value for the prediction. Use this to check how different the results are (score and
    predictions).
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    for c in list(df.columns):
        df = fix_outliers(df, c)
        df = fix_nans(df, c)
        df[c] = normalize_column(df[c])

    ohe = generate_one_hot_encoder(df['species'])
    df = replace_with_one_hot_encoder(
        df, 'species', ohe, list(ohe.get_feature_names()))

    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    return simple_random_forest_regressor(X, y)


##############################################
# Implement all the below methods
# Don't install any other python package other than provided by python or in requirements.txt
##############################################
def random_forest_iris_dataset_again() -> Dict:
    """
    Run the result of the process iris again task of e_experimentation and discuss (1 sentence)
    the differences from the above results. Also, as above, use sepal_length as the label column and
    the one_hot_encoder to transform the categorical column into a usable format.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_iris_dataset_again('regression')
    X, y = df.iloc[:, :len(df.columns.values) -
                   1], df.iloc[:, len(df.columns.values)-1]

    # The results are not very differnent. The scores fluctuate. One difference from previous model is
    # that the target variable is not normalized in this function.
    return simple_random_forest_regressor(X, y)


def decision_tree_regressor(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Reimplement the method "simple_random_forest_regressor" but using
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.DecisionTreeRegressor.html
    Optional: also optimise the parameters of the model to maximise the R^2 score
    :param X: Input dataframe
    :param y: Label data
    :return: model, score and prediction of the test set
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    score = model.score(X_test, y_test)

    return dict(model=model, score=score, test_prediction=y_predict)


def train_iris_dataset_again() -> Dict:
    """
    Run the result of the process iris again task of e_experimentation, but now using the
    decision tree regressor AND random_forest regressor. Return the one with highest R^2.
    Use the same label column and one hot encoding logic as before.
    Discuss (1 sentence) what you found different between the results.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_iris_dataset_again('regression')
    X, y = df.iloc[:, :len(df.columns.values) -
                   1], df.iloc[:, len(df.columns.values)-1]

    dt_model = decision_tree_regressor(X, y)
    rf_model = simple_random_forest_regressor(X, y)

    # Observation: After several runs, it was observed that RandomForestRegressor
    # outperformed DecisionTreeRegressor 9 out of 10 times
    if dt_model.get('score') > rf_model.get('score'):
        return dt_model

    return rf_model


def train_amazon_video_game() -> Dict:
    """
    Run the result of the amazon dataset task of e_experimentation using the
    decision tree regressor AND random_forest regressor. Return the one with highest R^2.
    The Label column is the count column
    Discuss (1 sentence) what you found different between the results.
    In one sentence, why is the score different (or the same) compared to the iris score?
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_amazon_video_game_dataset()

    X, y = df.iloc[:, :len(df.columns.values) -
                   1], df.iloc[:, len(df.columns.values)-1]

    dt_model = decision_tree_regressor(X, y)
    rf_model = simple_random_forest_regressor(X, y)

    # Observation 1: DecisionTreeRegressor and RandomForestRegressor are giving almost same scores
    # However, RandomForestRegressor is more consistent as compared to DecisionTreeRegressor
    # Observation 2: Amazon dataset score is better compared to iris because Iris dataset has more randomness
    # in their target variable sepal_length as compared to count in Amazon dataset
    if dt_model.get('score') > rf_model.get('score'):
        return dt_model

    return rf_model


def train_life_expectancy() -> Dict:
    """
    Do the same as the previous task with the result of the life expectancy task of e_experimentation.
    The label column is the value column. Remember to convert drop columns you think are useless for
    the machine learning (say why you think so) and convert the remaining categorical columns with one_hot_encoding.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_life_expectancy_dataset('regression')

    X, y = df.iloc[:, :len(df.columns.values) -
                   1], df.iloc[:, len(df.columns.values)-1]

    dt_model = decision_tree_regressor(X, y)
    rf_model = simple_random_forest_regressor(X, y)

    # Observation: DecisionTreeRegressor and RandomForestRegressor are giving almost same scores
    # However, RandomForestRegressor is more consistent as compared to DecisionTreeRegressor
    if dt_model.get('score') > rf_model.get('score'):
        return dt_model

    return rf_model


def your_choice() -> Dict:
    """
    Now choose one of the datasets included in the assignment1 (the raw one, before anything done to them)
    and decide for yourself a set of instructions to be done (similar to the e_experimentation tasks).
    Specify your goal (e.g. analyse the reviews of the amazon dataset), say what you did to try to achieve the goal
    and use one (or both) of the models above to help you answer that. Remember that these models are regression
    models, therefore it is useful only for numerical labels.
    We will not grade your result itself, but your decision-making and suppositions given the goal you decided.
    Use this as a small exercise of what you will do in the project.
    """
    # For this problem I will be predicting petal_length in the iris dataset.
    # Iris dataset has more numeric columns that could be predicted as compared to other datasets.
    # Since we have to start with the raw dataset, I will be doing the preprocessing here.
    df = read_dataset(Path('..', '..', 'iris.csv'))

    # Fixing outliers and nans
    # Random Forest/Decision Tree are tree-based models and hence do not require feature scaling.
    for c in list(df.columns):
        df = fix_outliers(df, c)
        df = fix_nans(df, c)

    # One-Hot-Encoding categorical variable - species
    ohe = generate_one_hot_encoder(df['species'])
    df = replace_with_one_hot_encoder(
        df, 'species', ohe, list(ohe.get_feature_names()))

    df = move_target_col_to_end(df, 'petal_length')

    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    # Choosing Random Forest since it has lower standard deviation and good accuracy
    # R^2 score was found to be 0.9781
    return simple_random_forest_regressor(X, y)


if __name__ == "__main__":
    assert simple_random_forest_on_iris() is not None
    assert reusing_code_random_forest_on_iris() is not None
    assert random_forest_iris_dataset_again() is not None
    assert train_iris_dataset_again() is not None
    assert train_amazon_video_game() is not None
    assert train_life_expectancy() is not None
    assert your_choice() is not None
