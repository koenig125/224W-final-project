"""
Script to run spectral clustering on the BIOSNAP Leeds Butterfly Dataset with 
Network Enhancement. Provides classification results (accuracy and confusion 
matrix) based on majority-label predictions from the clusters identified.
"""

import networkx as nx
from sklearn.cluster import SpectralClustering

import utils


def spectral_clustering(graph):
    """
    Compute graph partition using spectral clustering algorithm.

    :param - graph: nx.Graph object representing butterfly similarity network
    return: list where index i holds the community to which node with id i is assigned
    """
    adjacency_matrix = nx.to_numpy_matrix(graph)
    sc = SpectralClustering(n_clusters=10, n_init=1000, affinity='precomputed', assign_labels='kmeans')
    sc.fit(adjacency_matrix)
    return sc.labels_


def main():
    graph = utils.load_graph()
    position = utils.get_positions(graph)
    utils.make_dir('images/spectral')

    true_communities = utils.get_labels(graph, list(graph.nodes))
    utils.plot_communities(graph, position, true_communities, labels=True, title='Butterfly Similarity Network - True Communities', path='images/spectral/communities_true.png')

    node_assignments = spectral_clustering(graph)
    nodes_to_communities = {k:v for (k,v) in zip(range(len(node_assignments)), node_assignments)}
    communities = utils.group_communities(nodes_to_communities)
    utils.plot_communities(graph, position, communities, labels=False, title='Butterfly Similarity Network - Spectral Communities', path='images/spectral/communities_spectral.png')

    graph_nodes = sorted(list(graph.nodes))
    predictions = utils.predict_majority_class(graph, communities)
    preds = [predictions[n] for n in graph_nodes]
    labels = [graph.nodes[n]['label'] for n in graph_nodes]
    utils.accuracy(preds, labels)
    utils.confusion_matrix(preds, labels, 'Confusion Matrix - Spectral Clustering', 'images/spectral/cm_spectral.png')


if __name__=='__main__':
    main()
