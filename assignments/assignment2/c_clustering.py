from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.d_data_encoding import fix_outliers, fix_nans, normalize_column, \
    generate_one_hot_encoder, replace_with_one_hot_encoder, generate_label_encoder, replace_with_label_encoder
from assignments.assignment1.e_experimentation import process_iris_dataset, process_amazon_video_game_dataset

"""
Clustering is a non-supervised form of machine learning. It uses unlabeled data
 through a given method, returns the similarity/dissimilarity between rows of the data.
 See https://scikit-learn.org/stable/modules/clustering.html for an overview of methods in sklearn.
"""


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
def simple_k_means(X: pd.DataFrame, n_clusters=3, score_metric='euclidean') -> Dict:
    model = KMeans(n_clusters=n_clusters)
    clusters = model.fit_transform(X)

    # There are many methods of deciding a score of a cluster model. Here is one example:
    score = metrics.silhouette_score(X, model.labels_, metric=score_metric)
    return dict(model=model, score=score, clusters=clusters)


def iris_clusters() -> Dict:
    """
    Let's use the iris dataset and clusterise it:
    """
    df = pd.read_csv(Path('..', '..', 'iris.csv'))
    for c in list(df.columns):
        df = fix_outliers(df, c)
        df = fix_nans(df, c)
        df[c] = normalize_column(df[c])

    # Let's generate the clusters considering only the numeric columns first
    no_species_column = simple_k_means(df.iloc[:, :4])

    ohe = generate_one_hot_encoder(df['species'])
    df_ohe = replace_with_one_hot_encoder(df, 'species', ohe, list(ohe.get_feature_names()))

    # Notice that here I have binary columns, but I am using euclidean distance to do the clustering AND score evaluation
    # This is pretty bad
    no_binary_distance_clusters = simple_k_means(df_ohe)

    # Finally, lets use just a label encoder for the species.
    # It is still bad to change the labels to numbers directly because the distances between them does not make sense
    le = generate_label_encoder(df['species'])
    df_le = replace_with_label_encoder(df, 'species', le)
    labeled_encoded_clusters = simple_k_means(df_le)

    # See the result for yourself:
    print(no_species_column['score'], no_binary_distance_clusters['score'], labeled_encoded_clusters['score'])
    ret = no_species_column
    if no_binary_distance_clusters['score'] > ret['score']:
        ret = no_binary_distance_clusters
    if labeled_encoded_clusters['score'] > ret['score']:
        ret = labeled_encoded_clusters
    return ret


##############################################
# Implement all the below methods
# Don't install any other python package other than provided by python or in requirements.txt
##############################################
def custom_clustering(X: pd.DataFrame) -> Dict:
    """
    As you saw before, it is much harder to apply the right distance metrics. Take a look at:
    https://scikit-learn.org/stable/modules/clustering.html
    and check the metric used for each implementation. You will notice that suppositions were made,
    which makes harder to apply these clustering algorithms as-is due to the metrics used.
    Also go into each and notice that some of them there is a way to choose a distance/similarity/affinity metric.
    You don't need to check how each technique is implemented (code/math), but do use the information from the clustering
    lecture and check the parameters of the method (especially if there is any distance metric available among them).
    Chose one of them which is, in your opinion, generic enough, and justify your choice with a comment in the code (1 sentence).
    The return of this method should be the model, a score (e.g. silhouette
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) and the result of clustering the
    input dataset.
    """
    return dict(model=None, score=None, clusters=None)


def cluster_iris_dataset_again() -> Dict:
    """
    Run the result of the process iris again task of e_experimentation through the custom_clustering and discuss (3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    return dict(model=None, score=None, clusters=None)


def cluster_amazon_video_game() -> Dict:
    """
    Run the result of the process amazon_video_game task of e_experimentation through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    return dict(model=None, score=None, clusters=None)


def cluster_amazon_video_game_again() -> Dict:
    """
    Run the result of the process amazon_video_game_again task of e_experimentation through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    return dict(model=None, score=None, clusters=None)


def cluster_life_expectancy() -> Dict:
    """
    Run the result of the process life_expectancy task of e_experimentation through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    return dict(model=None, score=None, clusters=None)


if __name__ == "__main__":
    iris_clusters()
    assert cluster_iris_dataset_again() is not None
    assert cluster_amazon_video_game() is not None
    assert cluster_amazon_video_game_again() is not None
    assert cluster_life_expectancy() is not None
