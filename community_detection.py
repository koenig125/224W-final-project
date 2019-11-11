"""
Script to run Louvain community detection algorithm on the BIOSNAP Leeds Butterfly 
Dataset with Network Enhancement. Provides classification results (accuracy and 
confusion matrix) based on majority-label predictions from the communities 
identified by Louvain. Also plots communities identified by the algorithm.
"""

import sklearn.metrics as metrics
from community import best_partition as louvain

import utils


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


def report_classification_results(predictions, labels):
    """
    Calculate classification accuracy and confusion matrix given predictions and labels.

    :param - predictions: list for which index i holds the prediction for node i
    :param - labels: list for which index i holds ground truth label for node i
    """
    accuracy = metrics.accuracy_score(labels, predictions)
    print('Majority Label Classification Accuracy: {0:.5f}'.format(accuracy))
    confusion_matrix = metrics.confusion_matrix(labels, predictions)
    utils.plot_heatmap(confusion_matrix, 'Confusion Matrix - Majority Label Predictions from Louvain Communities', 'images/cm_louvain.png', 'Predicted label', 'True label')


def main():
    graph = utils.load_graph()
    position = utils.get_positions(graph)

    true_communities = get_labels(graph, list(graph.nodes))
    utils.plot_communities(graph, position, true_communities, labels=True, title='Butterfly Similarity Network - True Communities', path='images/communities_true.png')

    communities = partition(graph)
    utils.plot_communities(graph, position, communities, labels=False, title='Butterfly Similarity Network - Louvain Communities', path='images/communities_louvain.png')

    graph_nodes = sorted(list(graph.nodes))
    predictions = predict(graph, communities)
    preds = [predictions[n] for n in graph_nodes]
    labels = [graph.nodes[n]['label'] for n in graph_nodes]
    report_classification_results(preds, labels)


if __name__=='__main__':
    main()
