from typing import Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.widgets import Button, Slider

import sys
if "/home/aamir/dal/sem3/va/assignments/asakhan" not in sys.path:
    sys.path.append("/home/aamir/dal/sem3/va/assignments/asakhan")

from assignments.assignment3 import a_libraries, b_simple_usages
from assignments.assignment2.c_clustering import simple_k_means
from assignments.assignment1.d_data_encoding import fix_outliers, fix_nans, normalize_column

###############
# Interactivity in visualizations is challenging due to limitations and clunkiness of libraries.
# For example, some libraries works well in Jupyter Notebooks, but then the code makes barely any sense and becomes hard to change/update,
# defeating the purpose of using Jupyter notebooks in the first place, and other libraries provide a window of their own, but
# they are very tied to the running code, and far from the experience you'd expect from a proper visual analytics webapp
#
# We will try out some approaches to exercise in this file, but the next file will give you the proper tooling to make a
# well-rounded and efficient code for visual interactions.
##############################################
# Example(s). Read the comments in the following method(s)
##############################################


def matplotlib_simple_example():
    """
    Using the same logic from before, we can add sliders or buttons to select options for interactivity.
    Matplotlib is limited to only a few options, but do take a look, since they are useful for fast prototyping and analysis

    In case you are using PyCharm, I suggest you to uncheck the 'Show plots in tool window'
    to be able to see and interact with the buttons defined below.
    This example comes from https://matplotlib.org/3.1.1/gallery/widgets/buttons.html
    """
    freqs = np.arange(2, 20, 3)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    t = np.arange(0.0, 1.0, 0.001)
    s = np.sin(2 * np.pi * freqs[0] * t)
    l, = plt.plot(t, s, lw=2)

    class Index(object):
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    return fig, ax


def matplotlib_simple_example2():
    """
    Here is another example, which also has a slider and simplifies a bit the callbacks
    """
    data = np.random.rand(10, 5)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.bar(np.arange(10).astype(str).tolist(), data[:, 0])

    class Index(object):
        ind = 0
        multiplier = 1

        def change_data(self, event, i):
            self.ind = np.clip(self.ind + i, 0, data.shape[1] - 1)
            ax.clear()
            ax.bar(np.arange(10).astype(str).tolist(), data[:, self.ind] * self.multiplier)
            plt.draw()

        def change_multiplier(self, value):
            self.multiplier = value
            ax.clear()
            ax.bar(np.arange(10).astype(str).tolist(), data[:, self.ind] * self.multiplier)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.1, 0.05, 0.12, 0.075])
    axnext = plt.axes([0.23, 0.05, 0.12, 0.075])
    axslider = plt.axes([0.55, 0.1, 0.35, 0.03])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(lambda event: callback.change_data(event, 1))
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(lambda event: callback.change_data(event, -1))
    slider = Slider(axslider, 'multiplier', 1, 10, 1)
    slider.on_changed(callback.change_multiplier)

    return fig, ax


def plotly_slider_example():
    """
    Here is a simple example from https://plotly.com/python/sliders/ of how to include a slider in plotly
    Notice how similar it already is to GapMinder!
    """
    df = px.data.gapminder()
    fig = px.scatter(df, x="gdpPercap", y="lifeExp",
                     animation_frame="year",  # set which column makes the animation though a slider
                     size="pop",
                     color="continent",
                     hover_name="country",
                     log_x=True,
                     size_max=55,
                     range_x=[100, 100000],
                     range_y=[25, 90])

    fig["layout"].pop("updatemenus")  # optional, drop animation buttons

    return fig


def plotly_button_example():
    """
    To have buttons, plotly requires us to use go (and not px) to generate the graph.
    The button options are very restrictive, since all they can do is change a parameter from the go graph.
    In the example below, it changes the "mode" value of the graph (between lines and scatter)
    The code is a modified code example taken from https://plotly.com/python/custom-buttons/
    """
    x = np.random.rand(50) * np.random.randint(-10, 10)
    y = np.random.rand(50) * np.random.randint(-10, 10)

    fig = go.Figure()

    # Add surface trace
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(type="buttons",
                 direction="left",
                 buttons=[
                     dict(
                         label="line",  # just the name of the button
                         method="update",  # This is the method of update (check https://plotly.com/python/custom-buttons/)
                         args=[{"mode": "markers"}],  # This is the value being updated in the visualization
                     ), dict(
                         label="scatter",  # just the name of the button
                         method="update",  # This is the method of update (check https://plotly.com/python/custom-buttons/)
                         args=[{"mode": "line"}],  # This is the value being updated in the visualization
                     )
                 ],
                 pad={"r": 10, "t": 10}, showactive=True, x=0.11, xanchor="left", y=1.1, yanchor="top"  # Layout-related values
                 ),
        ]
    )

    fig.show()
    return fig


##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
##############################################
def matplotlib_interactivity():
    """
    Do an interactive matplotlib plot where I can select which visualization I want.
    Make either a slider, a dropdown or several buttons and make so each option gives me a different visualization from
    the matplotlib figures of b_simple_usages. Return just the resulting fig as is done in plotly_slider_example.
    """
    # As per the discussion on Teams, we can use the data from a_libraries.py for different plots. 
    # We are not using b_simple_usages functions because changing figures on click in matplotlib doesn't work
    # That is why we have to plot/draw figures on click.
    x = np.random.rand(50) * np.random.randint(-10, 10)
    y_arr = np.arange(start=0, stop=x.shape[0])
    matrix_2D = np.random.rand(10, 10) * np.random.randint(-10, 10)

    fig, ax = a_libraries.matplotlib_bar_chart(x)
    plt.subplots_adjust(bottom=0.2)

    class Index(object):

        def bar(self, event):
            ax.clear()
            ax.bar(y_arr, x)
            plt.draw()

        def pie(self, event):
            ax.clear()
            ax.pie(x)
            plt.draw()

        def histogram(self, event):
            ax.clear()
            ax.hist(x, 10)
            plt.draw()

        def heatmap(self, event):
            ax.clear()
            ax.imshow(matrix_2D)
            plt.draw()

    callback = Index()

    ax_bar = plt.axes([0.48, 0.05, 0.1, 0.075])
    ax_pie = plt.axes([0.59, 0.05, 0.1, 0.075])
    ax_hist = plt.axes([0.7, 0.05, 0.1, 0.075])
    ax_heat = plt.axes([0.81, 0.05, 0.1, 0.075])

    bbar = Button(ax_bar, 'Bar')
    bbar.on_clicked(callback.bar)

    bpie = Button(ax_pie, 'Pie')
    bpie.on_clicked(callback.pie)

    bhist = Button(ax_hist, 'Hist')
    bhist.on_clicked(callback.histogram)

    bheat = Button(ax_heat, 'Heat')
    bheat.on_clicked(callback.heatmap)

    return fig


def matplotlib_cluster_interactivity():
    """
    Do an interactive matplotlib plot where I can select how many clusters I want to train from.
    Use iris dataset (just numeric columns) and k-means (feel free to reuse as/c_clustering if you wish).
    The slider (or dropdown) should range from 2 to 10. Return just the resulting fig.
    """
    # I am plotting clusters with x as sepal_width and y as sepal_length. 
    # Clusters will be represented by different colors
    df = pd.read_csv(Path('..', '..', 'iris.csv'))
    for c in list(df.columns):
        df = fix_outliers(df, c)
        df = fix_nans(df, c)
        df[c] = normalize_column(df[c])


    # By default the number of clusters is 2
    model_data = simple_k_means(df.iloc[:, :4], n_clusters=2)

    fig, ax = plt.subplots()
    ax.scatter(x=df['sepal_width'], y=df['sepal_length'], c=model_data['model'].labels_, cmap='gist_rainbow')
    plt.subplots_adjust(bottom=0.2)

    class Index(object):

        def update_cluster_graph(self, value):
            model_data = simple_k_means(df.iloc[:, :4], n_clusters=value)
            ax.clear()
            ax.scatter(x=df['sepal_width'], y=df['sepal_length'], c=model_data['model'].labels_, cmap='gist_rainbow')
            plt.draw()

    callback = Index()

    ax_clusters_slider = plt.axes([0.55, 0.1, 0.35, 0.03])

    slider = Slider(ax_clusters_slider, 'n_clusters', 2, 10, valstep=1)
    slider.on_changed(callback.update_cluster_graph)

    return fig


def plotly_interactivity():
    """
    Do a plotly graph with all plotly 6 figs from b_simple_usages, and make 6 buttons (one for each fig).
    Change the displayed graph depending on which button I click. Return just the resulting fig.
    """
    return None


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    fig_m_i = matplotlib_interactivity()
    fig_m_ci = matplotlib_cluster_interactivity()
    fig_p =  plotly_interactivity()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # matplotlib_simple_example()[0].show()
    # matplotlib_simple_example2()[0].show()
    # plotly_slider_example().show()
    # plotly_button_example().show()
    # fig_m_i.show()
    # fig_m_ci.show()
    # fig_p.show()
