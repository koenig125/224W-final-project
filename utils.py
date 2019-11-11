import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib import cm


# Species names for the Leeds Butterfly Dataset.
# Additional information about this dataset here:
# http://www.josiahwang.com/dataset/leedsbutterfly/README.txt
labels_to_species = {
    1: 'Danaus plexippus',
    2: 'Heliconius charitonius',
    3: 'Heliconius erato',
    4: 'Junonia coenia',
    5: 'Lycaena phlaeas',
    6: 'Nymphalis antiopa',
    7: 'Papilio cresphontes',
    8: 'Pieris rapae',
    9: 'Vanessa atalanta',
    10: 'Vanessa cardui'
}

# Butterfly similarity network obtained from BIOSNAP. Nodes 
# represent butterflies (organisms) and edges represent visual 
# similarities between the organisms. Additional information here: 
# https://snap.stanford.edu/biodata/datasets/10029/10029-SS-Butterfly.html
edgefile = open("data/dataset/SS-Butterfly_weights.tsv", 'rb')
nodefile = open('data/dataset/SS-Butterfly_labels.tsv', 'r')


def load_graph():
    """
    Load butterfly similarity network using networkx graph representation.

    return: nx.Graph object representing the butterfly similarity network
    """
    # Create graph from weighted edge list.
    G = nx.read_weighted_edgelist(edgefile, '#', '\t', nx.Graph(), nodetype=int)

    # Add class labels as node attributes.
    for line in nodefile:
        if line[0] == '#': continue # skip comments
        node_id, label = line[:-1].split('\t')
        node_id, label = int(node_id), int(label)
        G.nodes[node_id]['label'] = label
    return G


def make_dir(dir_name):
    '''
    Creates a directory if it doesn't yet exist.
    
    :param - dir_name: Name of directory to be created.
    '''
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, dir_name + '/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)


def plot_heatmap(matrix, title, path, xlabel=None, ylabel=None):
    """
    Plots the provided matrix as a heatmap.

    :param - matrix: 2D numpy array representing values to plot in heatmap
    :param - title: title for the plot
    :param - path: save path for plot
    :param - xlabel: label for x-axis
    :param - ylabel: label for y-axis
    """
    plt.close('all')
    species = list(labels_to_species.values())
    df_cm = pd.DataFrame(matrix, species, species)
    _ = plt.figure(figsize=(10, 7))
    heatmap = sns.heatmap(df_cm, annot=True, cmap=sns.cm.rocket_r)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    make_dir('images')
    plt.savefig(path)


def get_positions(graph):
    """
    Calculate the positions to be used in visualizing graph.

    :param - graph: nx.Graph object representing butterfly similarity network
    return: dictionary of positions keyed by node
    """
    return nx.spring_layout(graph)


def plot_communities(graph, pos, communities, labels=False, cmap=cm.get_cmap('jet'), title=None, path=None):
    """
    Plot nodes in graph with coloring based on node assignments to communities.

    :param - graph: nx.Graph object representing butterfly similarity network
    :param - communities: dictionary mapping communities to nodes in those communities
    :param - labels: boolean indicating whether or not species labels should be applied
    :param - cmap: matplotlib colormap for mapping node classes to colors
    :param - title: title for the plot
    :param - path: save path for plot
    """
    plt.close('all')
    _ = plt.figure(figsize=(10, 7))
    for i, (c, nodes) in enumerate(communities.items()):
        c_index = int((i + 1) / len(communities) * 255)
        color = cmap(c_index)[:3] # 3 values for rgb
        label = labels_to_species[c] if labels else None
        nx.draw_networkx_nodes(graph, pos, nodes, node_size=30, label=label,
                                node_color=[color] * len(nodes), cmap=cmap)
    nx.draw_networkx_edges(graph, pos, alpha=0.05)
    if labels: plt.legend()
    plt.title(title)
    plt.tight_layout()
    make_dir('images')
    plt.savefig(path)
