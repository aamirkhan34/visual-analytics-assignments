from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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
    return None, None


def matplotlib_pie_chart() -> Tuple:
    """
    Create a pie chart where each piece of the chart has the number of columns which are numeric/categorical/binary
    from the output of a1/e_/process_life_expectancy_dataset
    """
    return None, None


def matplotlib_histogram() -> Tuple:
    """
    Build 4 histograms as subplots in one figure with the numeric values of the iris dataset
    """
    return None, None


def matplotlib_heatmap_chart() -> Tuple:
    """
    Remember a1/b_/pandas_profile? There is a heat map over there to analyse the correlation among columns.
    Use the pearson correlation (e.g. https://docs.scipy.org/doc/scipy-1.5.3/reference/generated/scipy.stats.pearsonr.html)
    to calculate the correlation between two numeric columns and show that as a heat map. Use the iris dataset.
    """
    return None, None


# There are many other possibilities. Please, do check the documentation and examples so you
# may have a good breadth of tools for future work (in assignments, projects, and your own career)
###################################
# Once again, for ALL methods return the fig and ax of matplotlib or fig from plotly!


def plotly_scatter_plot_chart():
    """
    Use the result of a2/c_clustering/cluster_iris_dataset_again() as the color of a scatterplot made from the original (unprocessed)
    iris dataset. Choose among the numeric values to be the x and y coordinates.
    """
    return None


def plotly_bar_plot_chart():
    """
    Use the result of a2/c_clustering/cluster_iris_dataset_again() and use x as 3 groups of bars (one for each iris species)
    and each group has multiple bars, one for each cluster, with y as the count of instances in the specific cluster/species combination.
    The grouped bar chart is like https://plotly.com/python/bar-charts/#grouped-bar-chart (search for the grouped bar chart visualization)
    """
    return None


def plotly_polar_scatterplot_chart():
    """
    Do something similar to a1/e_/process_life_expectancy_dataset, but don't drop the latitude and longitude.
    Use these two values to figure out the theta to plot values as a compass (example: https://plotly.com/python/polar-chart/).
    Each point should be one country and the radius should be thd value from the dataset (add up all years and feel free to ignore everything else)
    """
    return None


def plotly_table():
    """
    Show the data from a2/a_classification/your_choice() as a table
    See https://plotly.com/python/table/ for documentation
    """
    return None


def plotly_composite_line_bar():
    """
    Use the data from a1/e_/process_life_expectancy_dataset and show in a single graph on year on x and value on y where
    there are 5 line charts of 5 countries (you choose which) and one bar chart on the background with the total value of all 5
    countries added up.
    """
    return None


def plotly_map():
    """
    Use the data from a1/e_/process_life_expectancy_dataset on a plotly map (anyone will do)
    Examples: https://plotly.com/python/maps/, https://plotly.com/python/choropleth-maps/#using-builtin-country-and-state-geometries
    Use the value from the dataset of a specific year (e.g. 1900) to show as the color in the map
    """
    return None


def plotly_tree_map():
    """
    Use plotly's treemap to plot any data returned from any of a1/e_experimentation or a2 tasks
    Documentation: https://plotly.com/python/treemaps/
    """
    return None


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
    fig_p_treemap = plotly_map()

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
