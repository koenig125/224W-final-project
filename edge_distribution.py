"""
Script to calculate inter-species edge weight distribution for the BioSNAP butterfly
similarity network. Edge weights derived from Network Enhancement processing
(see https://snap.stanford.edu/biodata/datasets/10029/10029-SS-Butterfly.html).
"""

import numpy as np

import utils


def get_edge_distribution(graph):
    """
    Calculate the distribution of edge weights between nodes of different classes.

    :param - graph: nx.Graph object representing butterfly similarity network
    return: Row-normalized 2D numpy array for which each row and column represents 
    a species. Entry (i, j) is the percentage of the total edge weight for nodes  
    from species i that derives from edges to nodes from species j.
    """
    all_labels = set([graph.nodes[n]['label'] for n in graph.nodes])
    num_classes = len(all_labels)
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


def main():
    graph = utils.load_graph()
    edge_distribution = get_edge_distribution(graph)
    utils.plot_heatmap(edge_distribution, 'Row-Normalized Inter-Species Edge Weight Distribution Post Network Enhancement', 'images/edge_distribution.png')


if __name__=='__main__':
    main()
