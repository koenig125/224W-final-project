"""
Runs Louvain community detection algorithm on the Leeds Butterfly dataset
and provides species distribution analysis for detected communities. Also
provides classification results based on majority-label predictions from
the communities identified by Louvain.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from community import best_partition as louvain
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
    plt.savefig(path)


def get_edge_distribution(graph):
    """
    Calculate the distribution of edge weights between nodes of different classes.

    :param - graph: nx.Graph object representing butterfly similarity network
    return: Row-normalized 2D numpy array for which each row and column represents 
    a species. Entry (i, j) is the percentage of the total edge weight for nodes  
    from species i that derives from edges to nodes from species j.
    """
    num_classes = len(labels_to_species)
    edge_weights = np.zeros((num_classes, num_classes))
    for n in graph.nodes:
        l1 = graph.nodes[n]['label']
        for neighbor in graph.adj[n]:
            l2 = graph.nodes[neighbor]['label']
            weight = graph.adj[n][neighbor]['weight']
            edge_weights[l1 - 1][l2 - 1] += weight
    row_sums = edge_weights.sum(axis=1)
    normalized_distribution = edge_weights / row_sums[:, np.newaxis]
    return normalized_distribution


def partition(graph):
    """
    Compute graph partition to maximize modularity using Louvain algorithm.

    :param - graph: nx.Graph object representing butterfly similarity network
    return: dictionary mapping communities to nodes in those communities
    """
    communities = {}
    nodes_to_communities = louvain(graph, random_state=224)
    for n, c in nodes_to_communities.items():
        if c in communities:
            communities[c].append(n)
        else:
            communities[c] = [n]
    return communities


def report_species_distribution(graph, communities):
    """
    Print the species distribution of nodes in the communities provided.

    :param - graph: nx.Graph object representing butterfly similarity network
    :param - communities: dictionary mapping communities to nodes in those communities
    """
    for community, nodes in communities.items():
        print('COMMUNITY', community)
        print('# NODES:', len(nodes))
        species_distribution(graph, nodes)
        print()


def species_distribution(graph, nodes):
    """
    Print the distribution of species from nodes provided.

    :param - graph: nx.Graph object representing butterfly similarity network
    :param - nodes: list of node ids from graph
    """
    template = "{0:25} {1:5} {2:5}" 
    print(template.format("SPECIES", "COUNT", "PERCENT"))
    labels = get_labels(graph, nodes)
    for l, node_ids in labels.items():
        count = len(node_ids)
        percent = count / len(nodes)
        species = labels_to_species[l]
        print(template.format(species, count, '{0:.3f}'.format(percent)))


def get_labels(graph, nodes):
    """
    Get the labels of each node and group nodes by labels.

    :param - graph: nx.Graph object representing butterfly similarity network
    :param - nodes: list of node ids from graph
    return: dictionary mapping labels to nodes
    """
    labels = {}
    for node in nodes:
        l = graph.nodes[node]['label']
        if l in labels:
            labels[l].append(node)
        else:
            labels[l] = [node]
    return labels


def predict(graph, communities):
    """
    Make predictions using the majority class in each community as the predicted 
    label for every node in that community.

    :param - graph: nx.Graph object representing butterfly similarity network
    :param - communities: dictionary mapping communities to nodes in those communities
    """
    preds = {}
    for _, nodes in communities.items():
        labels = get_labels(graph, nodes)
        counts = [(l, len(nids)) for l, nids in labels.items()]
        counts.sort(key=lambda x: x[1], reverse=True)
        majority_label = counts[0][0]
        for n in nodes:
            preds[n] = majority_label
    return preds


def report_classification_results(predictions, labels):
    """
    Calculate classification accuracy and confusion matrix given predictions and labels.

    :param - predictions: list for which index i holds the prediction for node i
    :param - labels: list for which index i holds ground truth label for node i
    """
    accuracy = metrics.accuracy_score(labels, predictions)
    print('Majority Label Classification Accuracy: {0:.5f}'.format(accuracy))
    cm = metrics.confusion_matrix(labels, predictions)
    plot_heatmap(cm, 'Confusion Matrix - Majority Label Predictions from Louvain Communities', 'images/confusion_matrix.png', 'Predicted label', 'True label')


def plot_heatmap(matrix, title, path, xlabel=None, ylabel=None):
    """
    Plots the provided matrix as a heatmap using seaborn graphing library.

    :param - cm: confusion matrix, as returned by sklearn.metrics.confusion_matrix
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
    plt.savefig(path)


def main():
    graph = load_graph()
    position = nx.spring_layout(graph)
    
    edge_distribution = get_edge_distribution(graph)
    plot_heatmap(edge_distribution, 'Row-Normalized Inter-Species Edge Weight Distribution Post Network Enhancement', 'images/edge_distribution.png')

    true_communities = get_labels(graph, list(graph.nodes))
    plot_communities(graph, position, true_communities, labels=True, title='Butterfly Similarity Network - True Communities', path='images/true_communities.png')

    communities = partition(graph)
    plot_communities(graph, position, communities, labels=False, title='Butterfly Similarity Network - Louvain Communities', path='images/louvain_communities.png')

    graph_nodes = sorted(list(graph.nodes))
    predictions = predict(graph, communities)
    preds = [predictions[n] for n in graph_nodes]
    labels = [graph.nodes[n]['label'] for n in graph_nodes]
    report_classification_results(preds, labels)


if __name__=='__main__':
    main()
