import math
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.b_data_profile import get_text_categorical_columns
from assignments.assignment1.d_data_encoding import generate_label_encoder, replace_with_label_encoder, fix_outliers, fix_nans, normalize_column, generate_one_hot_encoder, replace_with_one_hot_encoder
from assignments.assignment1.e_experimentation import process_iris_dataset, process_iris_dataset_again, process_amazon_video_game_dataset_again, process_life_expectancy_dataset, move_target_col_to_end

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

    # If necessary, change the n_estimators, max_depth and max_leaf_nodes in the below method to accelerate the model training,
    # but don't forget to comment why you did and any consequences of setting them!
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    # Use this line to get the prediction from the model
    y_predict = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    return dict(model=model, accuracy=accuracy, test_prediction=y_predict)


def large_scale_random_forest_classifier(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to create and train a random forest classifier on a large dataset
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    chunk_size = 120000

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    no_of_batches = math.ceil(X_train.shape[0]/chunk_size)

    # Reduced the number of estimators to 10 for faster and more efficient fitting.
    # It will load only 10 trees in ensemble. I have also set warm_start to True so that
    # the model can incrementally fit on batches. Also setting max_depth to 6 so that the
    # Trees don't overfit and generalize better.
    model = RandomForestClassifier(
        n_estimators=10, warm_start=True, max_depth=6)

    y_train_chunks = np.array_split(y_train, no_of_batches)
    estimators = 10

    for idx, arr in enumerate(np.array_split(X_train, no_of_batches)):
        if idx > 0:
            # Incrementing estimators in order to fit new data because the old estimators are untouched once trained
            estimators += 10
            model.set_params(n_estimators=estimators)

        model.fit(arr, y_train_chunks[idx])

    # Use this line to get the prediction from the model
    y_predict = predict_on_large_dataset(model, X_test, 10000)
    accuracy = accuracy_score(y_test, y_predict)

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
    y_encoded = replace_with_label_encoder(
        y.to_frame(), column='species', le=le)

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
    df = process_iris_dataset_again('classification')
    X, y = df.iloc[:, :5], df.iloc[:, 5]
    le = generate_label_encoder(y)

    # Be careful to return a copy of the input with the changes, instead of changing inplace the inputs here!
    y_encoded = replace_with_label_encoder(
        y.to_frame(), column='species', le=le)

    # This model receives one additional feature: "large_sepal_length". After several iterations,
    # it was observed that the additional feature does not help the accuracy. In fact the previous model
    # is almost always slightly better than this model in terms of accuracy metric.
    return simple_random_forest_classifier(X, y_encoded['species'])


def decision_tree_classifier(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Reimplement the method "simple_random_forest_classifier" but using the technique we saw in class: decision trees
    (you can use sklearn to help you).
    Optional: also optimise the parameters of the model to maximise accuracy
    :param X: Input dataframe
    :param y: Label data
    :return: model, accuracy and prediction of the test set
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # If necessary, change the n_estimators, max_depth and max_leaf_nodes in the below method to accelerate the model training,
    # but don't forget to comment why you did and any consequences of setting them!
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    # Use this line to get the prediction from the model
    y_predict = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    return dict(model=model, accuracy=accuracy, test_prediction=y_predict)


def predict_on_large_dataset(model_x, test_X, chunk_size):
    no_of_batches = math.ceil(test_X.shape[0]/chunk_size)

    for idx, arr in enumerate(np.array_split(test_X, no_of_batches)):
        pred = model_x.predict(arr)
        if idx == 0:
            test_pred = pred.copy()
        else:
            test_pred = np.append(test_pred, pred, axis=0)

    return test_pred


def large_scale_decision_tree_classifier(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to create and train a decision tree classifier on a large dataset
    :param X: Input dataframe
    :param y: Label data
    :return: model, accuracy and prediction of the test set
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # If necessary, change the max_depth and max_leaf_nodes in the below method to accelerate the model training,
    # but don't forget to comment why you did and any consequences of setting them!
    # Setting max_depth to 6 so that the Trees don't overfit and generalize better.
    model = DecisionTreeClassifier(max_depth=6)
    model.fit(X_train, y_train)

    # Use this line to get the prediction from the model
    # y_predict = model.predict(X_test)
    y_predict = predict_on_large_dataset(model, X_test, 10000)

    accuracy = accuracy_score(y_test, y_predict)

    return dict(model=model, accuracy=accuracy, test_prediction=y_predict)


def train_iris_dataset_again() -> Dict:
    """
    Run the result of the iris dataset again task of e_experimentation using the
    decision_tree classifier AND random_forest classifier. Return the one with highest score.
    Discuss (1 sentence) what you found different between the two models and scores.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_iris_dataset_again("classification")
    X, y = df.iloc[:, :5], df.iloc[:, 5]
    le = generate_label_encoder(y)

    # Be careful to return a copy of the input with the changes, instead of changing inplace the inputs here!
    y_encoded = replace_with_label_encoder(
        y.to_frame(), column='species', le=le)

    # rf_model_acc = []
    # dt_model_acc = []
    # for i in range(20):
    rf_model = simple_random_forest_classifier(X, y_encoded['species'])
    dt_model = decision_tree_classifier(X, y_encoded['species'])
    # rf_model_acc.append(rf_model.get("accuracy"))
    # dt_model_acc.append(dt_model.get("accuracy"))

    # print(np.mean(rf_model_acc), np.mean(dt_model_acc),
    #       np.std(rf_model_acc), np.std(dt_model_acc))

    # Discussion: After 20 runs, it was observed that in most cases
    # RandomForests's accuracy was more consistent with lower standard deviation
    # On the other hand: Decision Tree is simpler and faster because it is single tree
    # compared to default 100 estimators (Trees) in the Random Forest model.
    # Standard deviation: RF = 0.0322, DT = 0.0336, Mean: RF = 0.9333, DT = 0.9311
    # I have commented the code for multiple runs for your reference
    if rf_model.get("accuracy") > dt_model.get("accuracy"):
        return rf_model

    return dt_model


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
    df = process_amazon_video_game_dataset_again()

    # Removing median column because average is already present and it is not affected by outliers.
    df.drop('median', inplace=True, axis=1)

    # Dropping rows that have lower counts. This is being done to improve the models prediction otherwise
    # the model will underfit. Currently, there are over 870000+ unique user labels out of 1324753 samples
    # The additional dropped rows will also speed up the training of ML models. I am only keeping the labels
    # that has at least 5 samples.
    df = df.loc[df['count'] >= 15]

    X, y = df.iloc[:, :4], df.iloc[:, 4]
    le = generate_label_encoder(y)

    y_encoded = replace_with_label_encoder(
        y.to_frame(), column='user', le=le)

    rf_model = large_scale_random_forest_classifier(X, y_encoded['user'])
    dt_model = large_scale_decision_tree_classifier(X, y_encoded['user'])

    # Discussion: After few runs, it was observed that in most cases
    # RandomForests's accuracy was more consistent with lower standard deviation like in previous case
    # On the other hand: Decision Tree is simpler and faster because it is single tree
    # compared to 10 estimators (Trees) in the Random Forest model.
    # Also the score is not as good as iris because of underfitting. There are not enough samples for the lables
    # On the other hand, iris has good distribution of labels that's why it outperforms Amazon model.
    if rf_model.get("accuracy") > dt_model.get("accuracy"):
        return rf_model

    return dt_model


def train_life_expectancy() -> Dict:
    """
    Do the same as the previous task with the result of the life expectancy task of e_experimentation.
    The label column is the column which has north/south. Remember to convert drop columns you think are useless for
    the machine learning (say why you think so) and convert the remaining categorical columns with one_hot_encoding.
    (check the c_regression examples to see example on how to do this one hot encoding)
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_life_expectancy_dataset('classification')

    X, y = df.iloc[:, :192], df.iloc[:, 192]
    ydf = y.to_frame()

    # LabelEncoding of latitude was already done in preprocessing
    rf_model = simple_random_forest_classifier(X, ydf['latitude'])
    dt_model = decision_tree_classifier(X, ydf['latitude'])

    # Discussion: After few runs, it was observed that in most cases
    # RandomForests's accuracy was more consistent with lower standard deviation like in previous case
    # On the other hand: Decision Tree is simpler and faster because it is single tree
    # compared to default 100 estimators (Trees) in the Random Forest model.
    if rf_model.get("accuracy") > dt_model.get("accuracy"):
        return rf_model

    return dt_model


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
    # For this problem I will be classifying Longitude labels. These labels are east/west indicating
    # eastern/western hemisphere. The other features would be similar to train_life_expectancy
    # Since we have to start with the raw dataset, I will be doing the preprocessing here.
    df_geo = read_dataset(Path('..', '..', 'geography.csv'))
    df_ley = read_dataset(Path('..', '..', 'life_expectancy_years.csv'))
    all_columns_ley = list(df_ley.columns.values)

    # fixing outlier and missing_values in ley dataset
    for column in all_columns_ley:
        df_ley = fix_outliers(df_ley, column)
        df_ley = fix_nans(df_ley, column)

    # Change format of ley - from wide to long
    # Ref: https://www.journaldev.com/33398/pandas-melt-unmelt-pivot-function
    df_ley_rearranged = pd.melt(
        df_ley, id_vars=all_columns_ley[0], var_name="year", value_vars=all_columns_ley[1:])

    # Merge ley and geo on country
    df_merged = df_ley_rearranged.merge(
        df_geo, left_on='country', right_on='name')

    # Drop other columns except
    df_merged = df_merged[['country',
                           'four_regions', 'value', 'Longitude']]
    df_merged = df_merged.rename(
        columns={"four_regions": "continent", "Longitude": "longitude"})

    # latitude: numerical to categorical (north vs south)
    df_merged.loc[(df_merged["longitude"] >= 0.0), "longitude_cat"] = "east"
    df_merged.loc[(df_merged["longitude"] < 0.0), "longitude_cat"] = "west"
    df_merged.drop("longitude", axis=1, inplace=True)
    df_merged = df_merged.rename(columns={"longitude_cat": "longitude"})
    le = generate_label_encoder(df_merged["longitude"])
    df_merged = replace_with_label_encoder(df_merged, "longitude", le)

    # One-Hot-Encode categorical columns except target column
    categorical_columns = get_text_categorical_columns(df_merged)
    for col in categorical_columns:
        ohe = generate_one_hot_encoder(df_merged[col])
        df_merged = replace_with_one_hot_encoder(
            df_merged, col, ohe, list(ohe.get_feature_names()))

    df_merged = move_target_col_to_end(df_merged, 'longitude')

    # Building models and returning the best model
    X, y = df_merged.iloc[:, :192], df_merged.iloc[:, 192]
    ydf = y.to_frame()

    # Choosing Random Forest since it has lower standard deviation and good accuracy
    rf_model = simple_random_forest_classifier(X, ydf['longitude'])

    return rf_model


if __name__ == "__main__":
    assert simple_random_forest_on_iris() is not None
    assert reusing_code_random_forest_on_iris() is not None
    assert random_forest_iris_dataset_again() is not None
    assert train_iris_dataset_again() is not None
    assert train_amazon_video_game_again() is not None
    assert train_life_expectancy() is not None
    assert your_choice() is not None
