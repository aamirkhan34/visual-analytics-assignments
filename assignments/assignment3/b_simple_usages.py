from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from pathlib import Path

import sys
if "/home/aamir/dal/sem3/va/assignments/asakhan" not in sys.path:
    sys.path.append("/home/aamir/dal/sem3/va/assignments/asakhan")

from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.b_data_profile import *
from assignments.assignment1.d_data_encoding import generate_label_encoder, replace_with_label_encoder, fix_outliers, fix_nans, normalize_column, generate_one_hot_encoder, replace_with_one_hot_encoder
from assignments.assignment1.e_experimentation import process_iris_dataset, process_iris_dataset_again, process_amazon_video_game_dataset_again, process_life_expectancy_dataset, move_target_col_to_end

from assignments.assignment2.c_clustering import cluster_iris_dataset_again
from assignments.assignment2.a_classification import your_choice

from assignments.assignment3 import a_libraries

##############################################
# In this file, we will use data and methods of previous assignments with visualization.
# But before you continue on, take some time to look on the internet about the many existing visualization types and their usages, for example:
# https://extremepresentation.typepad.com/blog/2006/09/choosing_a_good.html
# https://datavizcatalogue.com/
# https://plotly.com/python/
# https://www.tableau.com/learn/whitepapers/which-chart-or-graph-is-right-for-you
# Or just google "which visualization to use", and you'll find a near-infinite number of resources
#
# You may want to create a new visualization in the future, and for that I suggest using JavaScript and D3.js, but for the course, we will only
# use python and already available visualizations
##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
# For ALL methods return the fig and ax of matplotlib or fig from plotly!
##############################################
def matplotlib_bar_chart() -> Tuple:
    """
    Create a bar chart with a1/b_data_profile's get column max.
    Show the max of each numeric column from iris dataset as the bars
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    x = []

    for col in df.columns:
        try:
            max_val = get_column_max(df, col)
            x.append(max_val)
        except ValueError:
            pass
    
    fig, ax = a_libraries.matplotlib_bar_chart(np.array(x))

    return fig, ax


def matplotlib_pie_chart() -> Tuple:
    """
    Create a pie chart where each piece of the chart has the number of columns which are numeric/categorical/binary
    from the output of a1/e_/process_life_expectancy_dataset
    """
    df = process_life_expectancy_dataset("classification")
    num_cols = get_numeric_columns(df)
    bin_cols = get_binary_columns(df)
    text_cols = get_text_categorical_columns(df)

    x_arr = np.array([len(num_cols), len(bin_cols), len(text_cols)])
    # The plot only shows numeric columns because process_life_expectancy_dataset returned df only
    # contains numeric columns 
    fig, ax = a_libraries.matplotlib_pie_chart(x_arr)

    return fig, ax


def matplotlib_histogram() -> Tuple:
    """
    Build 4 histograms as subplots in one figure with the numeric values of the iris dataset
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    df.drop("species", axis=1, inplace=True)
    top_4_columns = list(df.columns)[:4]

    # Ref: https://stackoverflow.com/questions/31726643/how-do-i-get-multiple-subplots-in-matplotlib
    fig, ax = plt.subplots(nrows=2, ncols=2)

    c = 0
    for row in ax:
        for col in row:
            df_column = top_4_columns[c]
            col.hist(df[df_column].values)
            c = c + 1

    return fig, ax


def matplotlib_heatmap_chart() -> Tuple:
    """
    Remember a1/b_/pandas_profile? There is a heat map over there to analyse the correlation among columns.
    Use the pearson correlation (e.g. https://docs.scipy.org/doc/scipy-1.5.3/reference/generated/scipy.stats.pearsonr.html)
    to calculate the correlation between two numeric columns and show that as a heat map. Use the iris dataset.
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    df.drop("species", axis=1, inplace=True)
    # Default is pearson's correlation coefficient
    corr_df = df.corr()

    fig, ax = a_libraries.matplotlib_heatmap_chart(corr_df.values)

    return fig, ax


# There are many other possibilities. Please, do check the documentation and examples so you
# may have a good breadth of tools for future work (in assignments, projects, and your own career)
###################################
# Once again, for ALL methods return the fig and ax of matplotlib or fig from plotly!


def plotly_scatter_plot_chart():
    """
    Use the result of a2/c_clustering/cluster_iris_dataset_again() as the color of a scatterplot made from the original (unprocessed)
    iris dataset. Choose among the numeric values to be the x and y coordinates.
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))

    model_data = cluster_iris_dataset_again()
    df['clusters'] = model_data['clusters']

    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="clusters")

    return fig


def plotly_bar_plot_chart():
    """
    Use the result of a2/c_clustering/cluster_iris_dataset_again() and use x as 3 groups of bars (one for each iris species)
    and each group has multiple bars, one for each cluster, with y as the count of instances in the specific cluster/species combination.
    The grouped bar chart is like https://plotly.com/python/bar-charts/#grouped-bar-chart (search for the grouped bar chart visualization)
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))

    model_data = cluster_iris_dataset_again()
    df['clusters'] = model_data['clusters']

    # Species wise clusters count
    count_df = df.groupby(["species", "clusters"]).size().unstack(fill_value=0).stack().reset_index()
    count_df.columns = ["species", "clusters", "count"]
    count_df['clusters'] = count_df['clusters'].astype(str)

    fig = px.bar(count_df, x="species", color="clusters",
                y="count",
                barmode='group'
                )

    return fig


def plotly_polar_scatterplot_chart():
    """
    Do something similar to a1/e_/process_life_expectancy_dataset, but don't drop the latitude and longitude.
    Use these two values to figure out the theta to plot values as a compass (example: https://plotly.com/python/polar-chart/).
    Each point should be one country and the radius should be thd value from the dataset (add up all years and feel free to ignore everything else)
    """
    ley_df = process_life_expectancy_dataset("regression")
    geo_df = pd.read_csv(Path('..', '..', 'geography.csv'))

    ley_df = convert_ohe_columns_into_one(ley_df, "x0", "country")

    # Group by country and sum value across all the years
    ley_df = ley_df[["country", "value"]].groupby(["country"]).sum().reset_index()
    ley_df.head()

    # Merge latitude and logitude column to processed ley dataframe
    ley_df = pd.merge(ley_df, geo_df[["name", "Latitude", "Longitude"]], left_on="country", right_on="name")
    ley_df.drop("name", axis=1, inplace=True)

    # Latitude, Longitude to cartesian coordiantes
    # Ref: https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates
    EARTH_RADIUS = 6371
    ley_df['x'] = EARTH_RADIUS * np.cos(ley_df["Latitude"]) * np.cos(ley_df["Longitude"])
    ley_df['y'] = EARTH_RADIUS * np.cos(ley_df["Latitude"]) * np.sin(ley_df["Longitude"])

    # Calculating theta from cartesian coordinates
    # Ref: https://www.engineeringtoolbox.com/converting-cartesian-polar-coordinates-d_1347.html
    ley_df["theta"] = np.arctan2(ley_df["y"], ley_df["x"])

    # Plotting on polar coordinates
    fig = px.scatter_polar(ley_df, r="value", theta="theta", color="country")

    return fig


def plotly_table():
    """
    Show the data from a2/a_classification/your_choice() as a table
    See https://plotly.com/python/table/ for documentation
    """
    model_data = your_choice()
    model_data["test_prediction"] = list(model_data["test_prediction"])
    
    df = pd.DataFrame(model_data["test_prediction"], columns=["test_prediction"])
    for k,v in model_data.items():
        if k != "test_prediction":
            df[k] = str(v)

    fig = a_libraries.plotly_table(df)

    return fig


def plotly_composite_line_bar():
    """
    Use the data from a1/e_/process_life_expectancy_dataset and show in a single graph on year on x and value on y where
    there are 5 line charts of 5 countries (you choose which) and one bar chart on the background with the total value of all 5
    countries added up.
    """
    df = process_life_expectancy_dataset("regression")

    # Countries selected: India, Pakistan, United States, Canada, Brazil
    # Since the dataset is already one hot encoded, I will be restructuring it with new column called country
    country_columns = ["x0_Canada", "x0_United States", "x0_India", "x0_Pakistan","x0_Brazil"]

    # Selecting the above countries
    selected_df = df[(df[country_columns]).any(1)]

    # Filtering the required columns
    selected_df = selected_df[["year", "value"] + country_columns]

    # Restructuring columns
    for country in country_columns:
        selected_df.loc[selected_df[country] == 1, "country"] = country.lstrip("x0_")

    selected_df = selected_df[["country", "year", "value"]]

    # Bar chart - sum of all the country values by year
    bar_df = selected_df[["year", "value"]].groupby(["year"]).sum().reset_index()
    fig = px.bar(bar_df, x="year", y="value")

    # Line Charts - 5 line charts for each country by year
    for country in set(selected_df['country'].tolist()):
        country_df = selected_df[selected_df['country'] == country]
        fig.add_trace(go.Scatter(x = country_df['year'], y = country_df['value'], name=country))

    return fig

def convert_ohe_columns_into_one(xdf, prefix, new_column_name):
    ohe_columns = [col for col in xdf.columns if col.startswith(prefix+"_")]

    # Adding new column
    for ohe_col in ohe_columns:
        xdf.loc[xdf[ohe_col] == 1, new_column_name] = ohe_col.lstrip(prefix+"_")

    # Remove ohe columns
    xdf.drop(ohe_columns, axis=1, inplace=True)

    return xdf

def plotly_map():
    """
    Use the data from a1/e_/process_life_expectancy_dataset on a plotly map (anyone will do)
    Examples: https://plotly.com/python/maps/, https://plotly.com/python/choropleth-maps/#using-builtin-country-and-state-geometries
    Use the value from the dataset of a specific year (e.g. 1900) to show as the color in the map
    """
    df = process_life_expectancy_dataset("regression")

    selected_df = convert_ohe_columns_into_one(df, "x0", "country")

    # Choosing year 1800 for map plots
    selected_df = selected_df[selected_df["year"] == "1800"]

    # Plotting on Map
    fig = px.choropleth(selected_df, locations="country", locationmode="country names", color="value",
                        hover_name="country", color_continuous_scale = px.colors.sequential.Plasma)

    return fig


def plotly_tree_map():
    """
    Use plotly's treemap to plot any data returned from any of a1/e_experimentation or a2 tasks
    Documentation: https://plotly.com/python/treemaps/
    """
    df = process_life_expectancy_dataset("regression")

    # Drop latitude and year column
    df.drop(["latitude", "year"], axis=1, inplace=True)

    df = convert_ohe_columns_into_one(df, "x0", "country")
    df = convert_ohe_columns_into_one(df, "continent", "continent")

    tree_df = df.groupby(["continent", "country", "value"]).sum().reset_index()

    # Plotting Treemap with Continent as parent, country as child, and values representing total size country-wise
    fig = px.treemap(tree_df, path=['continent', 'country'], values='value')

    return fig

if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    fig_m_bc, _ = matplotlib_bar_chart()
    fig_m_pc, _ = matplotlib_pie_chart()
    fig_m_h, _ = matplotlib_histogram()
    fig_m_hc, _ = matplotlib_heatmap_chart()

    fig_p_s = plotly_scatter_plot_chart()
    fig_p_bpc = plotly_bar_plot_chart()
    fig_p_psc = plotly_polar_scatterplot_chart()
    fig_p_t = plotly_table()
    fig_p_clb = plotly_composite_line_bar()
    fig_p_map = plotly_map()
    fig_p_treemap = plotly_tree_map()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # fig_m_bc.show()
    # fig_m_pc.show()
    # fig_m_h.show()
    # fig_m_hc.show()
    #
    # fig_p_s.show()
    # fig_p_bpc.show()
    # fig_p_psc.show()
    # fig_p_t.show()
    # fig_p_clb.show()
    # fig_p_map.show()
    # fig_p_treemap.show()
