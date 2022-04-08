import ocel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import scaleogram as scg
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter


def plot_2d(X, row_ids, log):
    g_obj_types = ocel.get_object_types(log)
    g_obj = ocel.get_objects(log)
    colours = [np.random.rand(3,) for _ in range(len(g_obj_types))]

    colour_choice = [colours[g_obj_types.index(g_obj[obj_id]['ocel:type'])]
                     for obj_id in row_ids]

    fig, ax = plt.subplots()
    colour_handles = []
    for i, obj_type in enumerate(ocel.get_object_types(log)):
        colour_handles.append(mpatches.Patch(color=colours[i], label=obj_type))

    ax.scatter(X[:, 0], X[:, 1], c=colour_choice)
    plt.legend(handles=colour_handles)
    plt.show()


def plot_3d(X, row_ids, log):
    g_obj_types = ocel.get_object_types(log)
    g_obj = ocel.get_objects(log)
    colours = [np.random.rand(3,) for _ in range(len(g_obj_types))]

    colour_choice = [colours[g_obj_types.index(g_obj[obj_id]['ocel:type'])]
                     for obj_id in row_ids]

    fig = plt.figure()
    colour_handles = []
    for i, obj_type in enumerate(ocel.get_object_types(log)):
        colour_handles.append(mpatches.Patch(color=colours[i], label=obj_type))

    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colour_choice)
    plt.legend(handles=colour_handles)
    plt.show()


def plot_time_series_scatter(series):
    fig, ax = plt.subplots()
    ax.scatter(list(range(len(series))), series)
    plt.show()


def plot_time_series_wavelet(series, scaling=None):
    try:
        scg.cws(series)
    except Exception:
        pass


def plot_degree_graph(graph):
    plt.figure()
    data_xy = Counter([x[1] for x in nx.degree(graph)])
    plt.scatter(data_xy.keys(), data_xy.values())
    plt.show()

