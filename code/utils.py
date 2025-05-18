import platform

import cv2
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def get_video_capture():
    if platform.system() == "Darwin":  # macOS
        backend = cv2.CAP_AVFOUNDATION
    else:
        backend = None # idk, havent tested elsewhere
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        raise Exception('failed to open camera')
    return cap


def rgb2grey(im: np.ndarray):
    if (im is None) or (len(im.shape) == 2):
        return im
    return im[..., :3].dot([0.2989, 0.5870, 0.1140])


def get_appropriate_dims_for_ax_grid(n) -> tuple:
    """
    Minimize np.abs(nrows - ncols),
    subject to the constraints that:
        nrows, ncols are both ints
        nrows * ncols >= n

    return (nrows, ncols)
    """
    best_dims = None
    best_delta = np.inf
    for ncols in range(1, n):
    #  for ncols in range(np.ceil(n**.5), 0, -1): # need to test it before leaving this line uncommented
        nrows = np.ceil(n / ncols).astype(int)
        delta = np.abs(nrows - ncols)
        if delta < best_delta:
            best_delta = delta
            best_dims = (nrows, ncols)
    return best_dims


def normalize(im):
    """ scale im pixels to [0, 1] """
    try:
        im = im.astype('float')
    except AttributeError:
        pass
    try:
        im = im.to(torch.float32)
    except AttributeError:
        pass

    im = im - im.min() # zero min
    im = im / (im.max() + 1e-9) # unit max
    return im


def inspect(label, im):
    """ Print some basic image stats."""
    print()
    print(label + ':')
    print('shape:', im.shape)
    print('dtype:', im.dtype)

    try:
        print('min:', im.min())
        print('max:', im.max())
        print('mean:', im.mean())
        print('std:', im.std())
    except:
        pass

    print()



def set_pos_data(graph: nx.Graph, pos: dict = None):
    """ Set pos data for graph using the pos dict provided, or according to the
        kamada kawai layout.
    """
    if pos is None:
        pos = nx.get_node_attributes(graph, 'pos')
        if not all(n in pos for n in graph.nodes):
            pos = nx.kamada_kawai_layout(graph)
        else:
            return pos
    nx.set_node_attributes(graph, pos, 'pos')
    return pos




def plot(
    graph: nx.Graph,
    pos: dict = None,
    title=None,
    node_size=200,
    with_labels='deprecated',
    with_node_labels=True,
    with_edge_labels=True,
    ax=None,
    show_xy_ticks=False,
    show_borders=True,
    node_color='tab:blue',
    edge_color='black',
    node_transparency=0.5,
    edge_transparency=0.5,
    node_label_rotation=0,
    show_gridlines=True,
    gridlines_transparency=0.1,
    xlim=None,
    ylim=None,
):
    """ Draw a nx graph.
        This function wraps nx and plt calls with default values.

        graph: the nx graph to draw
        #  pos: a dict keyed by node whos value is a position in the xy plane
        title: the title of the plot
        node_size: scalar or a list of scalars
        with_node_labels: whether or not node labels should be drawn
        with_edge_labels: whether or not edge labels should be drawn
        ax: the Axes to plot on
        show_xy_ticks: whether or not to label the x and y axis
        show_borders: whether or not to show the borders of the plot
        node_color: color or list of colors
        edge_color: color or list of colors
        node_transparency: scalar in between 0 and 1
        edge_transparency: scalar in between 0 and 1
        node_label_rotation: degrees of rotation for the node labels
        show_gridlines: whether or not to show gridlines on the plot
        gridlines_transparency: scalar in between 0 and 1
        xlim: tuple
        ylim: tuple
    """
    if ax is None:
        fig, ax = plt.subplots()

    if with_labels != 'deprecated':
        warnings.warn(
            'with_labels` is deprecated in favor of `with_node_labels` or `with_edge_labels`'
        )
        with_node_labels = with_labels
        with_edge_labels = with_labels

    pos = set_pos_data(graph, pos)

    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        ax=ax,
        node_color=node_color,
        node_size=node_size,
        alpha=node_transparency,
    )
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        ax=ax,
        edge_color=edge_color,
        width=1.0,
        arrows=True,
        arrowsize=10,
        alpha=edge_transparency,
    )

    if with_node_labels:
        labels = nx.draw_networkx_labels(
            graph,
            pos,
            labels=nx.get_node_attributes(graph, 'node_labels')
            or {n: n
                for n in graph.nodes()},
            ax=ax,
        )
        if node_label_rotation:
            for label in labels.values():
                label.set_rotation(node_label_rotation)

    if with_edge_labels:
        labels = nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=nx.get_edge_attributes(graph, 'label'),
            ax=ax,
        )

    ax.set_title(title)

    if show_xy_ticks:
        ax.tick_params(left=True,
                       bottom=True,
                       labelleft=True,
                       labelbottom=True)

    if show_borders:
        ax.axis('on')
    else:
        ax.axis('off')

    if show_gridlines:
        ax.xaxis.grid(True, which='both', alpha=gridlines_transparency)
        ax.yaxis.grid(True, which='both', alpha=gridlines_transparency)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
