"""
Runs Louvain community detection algorithm on the Leeds Butterfly dataset
and provides species distribution analysis for detected communities. Also
provides classification results based on majority-label predictions from
the communities identified by Louvain.
"""

import numpy as np
import sklearn.metrics as metrics
from community import best_partition as louvain

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
    confusion_matrix = metrics.confusion_matrix(labels, predictions)
    utils.plot_heatmap(confusion_matrix, 'Confusion Matrix - Majority Label Predictions from Louvain Communities', 'images/confusion_matrix.png', 'Predicted label', 'True label')


def main():
    graph = utils.load_graph()
    position = utils.get_positions(graph)
    
    edge_distribution = get_edge_distribution(graph)
    utils.plot_heatmap(edge_distribution, 'Row-Normalized Inter-Species Edge Weight Distribution Post Network Enhancement', 'images/edge_distribution.png')

    true_communities = get_labels(graph, list(graph.nodes))
    utils.plot_communities(graph, position, true_communities, labels=True, title='Butterfly Similarity Network - True Communities', path='images/true_communities.png')

    communities = partition(graph)
    utils.plot_communities(graph, position, communities, labels=False, title='Butterfly Similarity Network - Louvain Communities', path='images/louvain_communities.png')

    graph_nodes = sorted(list(graph.nodes))
    predictions = predict(graph, communities)
    preds = [predictions[n] for n in graph_nodes]
    labels = [graph.nodes[n]['label'] for n in graph_nodes]
    report_classification_results(preds, labels)


if __name__=='__main__':
    main()
