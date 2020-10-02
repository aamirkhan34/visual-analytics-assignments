from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.d_data_encoding import generate_label_encoder, replace_with_label_encoder, fix_outliers, fix_nans, normalize_column
from assignments.assignment1.e_experimentation import process_iris_dataset, process_amazon_video_game_dataset_again

"""
Classification is a supervised form of machine learning. It uses labeled data, which is data with an expected
result available, and uses it to train a machine learning model to predict the said result. Classification
focuses in results of the categorical type.
"""


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
def simple_random_forest_classifier(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Simple method to create and train a random forest classifier
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)  # Use this line to get the prediction from the model
    accuracy = model.score(X_test, y_test)
    return dict(model=model, accuracy=accuracy, test_prediction=y_predict)


def simple_random_forest_on_iris() -> Dict:
    """
    Here I will run a classification on the iris dataset with random forest
    """
    df = pd.read_csv(Path('..', '..', 'iris.csv'))
    X, y = df.iloc[:, :4], df.iloc[:, 4]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return simple_random_forest_classifier(X, y_encoded)


def reusing_code_random_forest_on_iris() -> Dict:
    """
    Again I will run a classification on the iris dataset, but reusing
    the existing code from assignment1. Use this to check how different the results are (score and
    predictions).
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    for c in list(df.columns):
        # Notice that I am now passing though all columns.
        # If your code does not handle normalizing categorical columns, do so now (just return the unchanged column)
        df = fix_outliers(df, c)
        df = fix_nans(df, c)
        df[c] = normalize_column(df[c])

    X, y = df.iloc[:, :4], df.iloc[:, 4]
    le = generate_label_encoder(y)

    # Be careful to return a copy of the input with the changes, instead of changing inplace the inputs here!
    y_encoded = replace_with_label_encoder(y.toframe(), column='species', le=le)
    return simple_random_forest_classifier(X, y_encoded['species'])


def reusing_code_of_e_random_forest_on_iris() -> Dict:
    """
    Again I will run a classification on the iris dataset, but reusing
    the existing code from assignment1. Use this to check how different the results are (score and
    predictions).
    """
    df = process_iris_dataset()
    X, y = df.iloc[:, :4], df.iloc[:, 4]
    le = generate_label_encoder(y)
    y_encoded = replace_with_label_encoder(y.toframe(), column='species', le=le)
    return simple_random_forest_classifier(X, y_encoded['species'])


##############################################
# Implement all the below methods
# Don't install any other python package other than provided by python or in requirements.txt
##############################################
def random_forest_iris_dataset_again() -> Dict:
    """
    Run the result of the process iris again task of e_experimentation and discuss (1 sentence)
    the differences from the above results. Use the same random forest method.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    pass


def decision_tree_classifier(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Reimplement the method "simple_random_forest_classifier" but using the technique we saw in class: decision trees
    (you can use sklearn to help you).
    Optional: also optimise the parameters of the model to maximise accuracy
    :param X: Input dataframe
    :param y: Label data
    :return: model, accuracy and prediction of the test set
    """
    return dict(model=None, accuracy=None, test_prediction=None)


def train_iris_dataset_again() -> Dict:
    """
    Run the result of the iris dataset again task of e_experimentation using the
    decision_tree classifier AND random_forest classifier. Return the one with highest score.
    Discuss (1 sentence) what you found different between the two models and scores.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    return dict(model=None, accuracy=None, test_prediction=None)


def train_amazon_video_game_again() -> Dict:
    """
    Run the result of the amazon dataset again task of e_experimentation using the
    decision tree classifier AND random_forest classifier. Return the one with highest score.
    The Label column is the user column. Choose what you wish to do with the time column (drop, convert, etc)
    Discuss (1 sentence) what you found different between the results.
    In one sentence, why is the score worse than the iris score (or why is it not worse) in your opinion?
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    return dict(model=None, accuracy=None, test_prediction=None)


def train_life_expectancy() -> Dict:
    """
    Do the same as the previous task with the result of the life expectancy task of e_experimentation.
    The label column is the column which has north/south. Remember to convert drop columns you think are useless for
    the machine learning (say why you think so) and convert the remaining categorical columns with one_hot_encoding.
    (check the c_regression examples to see example on how to do this one hot encoding)
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    return dict(model=None, accuracy=None, test_prediction=None)


def your_choice() -> Dict:
    """
    Now choose one of the datasets included in the assignment1 (the raw one, before anything done to them)
    and decide for yourself a set of instructions to be done (similar to the e_experimentation tasks).
    Specify your goal (e.g. analyse the reviews of the amazon dataset), say what you did to try to achieve the goal
    and use one (or both) of the models above to help you answer that. Remember that these models are classification
    models, therefore it is useful only for categorical labels.
    We will not grade your result itself, but your decision-making and suppositions given the goal you decided.
    Use this as a small exercise of what you will do in the project.
    """
    pass


if __name__ == "__main__":
    assert simple_random_forest_on_iris() is not None
    assert reusing_code_random_forest_on_iris() is not None
    assert reusing_code_of_e_random_forest_on_iris() is not None
    assert random_forest_iris_dataset_again() is not None
    assert train_iris_dataset_again() is not None
    assert train_amazon_video_game_again() is not None
    assert train_life_expectancy() is not None
    assert your_choice() is not None
