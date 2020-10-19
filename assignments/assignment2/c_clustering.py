from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans, OPTICS, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.d_data_encoding import fix_outliers, fix_nans, normalize_column, \
    generate_one_hot_encoder, replace_with_one_hot_encoder, generate_label_encoder, replace_with_label_encoder
from assignments.assignment1.e_experimentation import process_iris_dataset_again, process_amazon_video_game_dataset, process_amazon_video_game_dataset_again, process_life_expectancy_dataset

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
    df_ohe = replace_with_one_hot_encoder(
        df, 'species', ohe, list(ohe.get_feature_names()))

    # Notice that here I have binary columns, but I am using euclidean distance to do the clustering AND score evaluation
    # This is pretty bad
    no_binary_distance_clusters = simple_k_means(df_ohe)

    # Finally, lets use just a label encoder for the species.
    # It is still bad to change the labels to numbers directly because the distances between them does not make sense
    le = generate_label_encoder(df['species'])
    df_le = replace_with_label_encoder(df, 'species', le)
    labeled_encoded_clusters = simple_k_means(df_le)

    # See the result for yourself:
    print(no_species_column['score'], no_binary_distance_clusters['score'],
          labeled_encoded_clusters['score'])
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
def custom_clustering(X: pd.DataFrame, epsilon=0.5, minimum_samples=5) -> Dict:
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
    # I have chosen DBSCAN algorithm. DBSCAN is not susceptibe to noise & outliers.
    # DBSCAN is efficient as well. # K-means is also generic but it is susceptible to outliers and noise.
    model = DBSCAN(eps=epsilon, min_samples=minimum_samples, n_jobs=-1)
    clusters = model.fit_predict(X)

    # Using silhoutte score
    score = metrics.silhouette_score(X, model.labels_, sample_size=10000)

    return dict(model=model, score=score, clusters=clusters)


def cluster_iris_dataset_again() -> Dict:
    """
    Run the result of the process iris again task of e_experimentation through the custom_clustering and discuss (3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    df = process_iris_dataset_again('clustering')

    # Discussion: From the output it was observed that the score was quite decent with the score of 0.7232
    # There were 5 clusters that were formed {0, 1, 2, 3, 4, -1}. The cluster -1 indicates the outliers
    # In general DBSCAN is better than K-means we do not need to set initial clusters and DBSCAN is not susceptible to outliers
    model_data = custom_clustering(df)

    return model_data


def cluster_amazon_video_game() -> Dict:
    """
    Run the result of the process amazon_video_game task of e_experimentation through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    df = process_amazon_video_game_dataset()
    df.drop('asin', axis=1, inplace=True)
    df = df.drop_duplicates()

    # Scale here and not in e_experimentation because b_regression file doesn't require scaling
    for c in list(df.columns):
        df[c] = normalize_column(df[c])

    # Removed asin column as it was categorical and after removing duplicates, there were over 50,000 unique users
    # Also, the length of samples equalled the number of unique asin.
    # Keeping asin didn't make any sense as we would want to cluster similar users based on their reviews and other features
    # Epsilon was chosen to be 0.01 as higher epsilon resulted in a single cluster only. No outliers were found.
    # Minimum samples was set to default value of 5 because larger minimum samples resulted in same score
    # Smaller minimum samples resulted in lower silhoutte score.
    # From the output it was observed that the scoren was bad with the score of 0.2282
    # There were 3 clusters that were formed {0, 1, 2, -1}. The cluster -1 indicates the outliers
    model_data = custom_clustering(df, epsilon=0.01)

    return model_data


def cluster_amazon_video_game_again() -> Dict:
    """
    Run the result of the process amazon_video_game_again task of e_experimentation through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    df = process_amazon_video_game_dataset_again()
    # Removing user column
    df.drop('user', axis=1, inplace=True)
    df = df.drop_duplicates()

    # Scale here and not in e_experimentation because b_regression file doesn't require scaling
    for c in list(df.columns):
        df[c] = normalize_column(df[c])

    # Removed user column as it was categorical and after removing duplicates, there were over 870,000 unique users
    # Also, the length of samples equalled the number of unique users.
    # Keeping users didn't make any sense as we would want to cluster similar users based on their reviews and other features
    # Epsilon was chosen to be 0.3 as higher epsilon resulted in a single cluster only. No outliers were found.
    # Minimum samples was set to default value of 5 because larger minimum samples resulted in a single cluster
    # Smaller minimum samples reslulted in lower silhoutte score.
    # From the output it was observed that the score bad with the score of 0.4715
    # There were 2 clusters that were formed {0, 1}. There were no outliers
    model_data = custom_clustering(df, epsilon=0.30)

    return model_data


def cluster_life_expectancy() -> Dict:
    """
    Run the result of the process life_expectancy task of e_experimentation through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    df = process_life_expectancy_dataset('clustering')

    model_data = custom_clustering(df, epsilon=0.3)
    # Discussion: From the output it was observed that the score was really good with the score of 0.8606
    # There were 187 clusters that were formed {0, 1, .. 186}. There were no outliers
    # In general DBSCAN is better than K-means we do not need to set initial clusters and DBSCAN is not susceptible to outliers

    return model_data


if __name__ == "__main__":
    iris_clusters()
    assert cluster_iris_dataset_again() is not None
    assert cluster_amazon_video_game() is not None
    assert cluster_amazon_video_game_again() is not None
    assert cluster_life_expectancy() is not None
