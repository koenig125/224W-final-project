"""
Script to run Louvain community detection algorithm on the BIOSNAP Leeds Butterfly 
Dataset with Network Enhancement. Provides classification results (accuracy and 
confusion matrix) based on majority-label predictions from the communities 
identified by Louvain. Also plots communities identified by the algorithm.
"""

import sklearn.metrics as metrics
from community import best_partition as louvain

import utils


def louvain_clustering(graph):
    """
    Compute graph partition to maximize modularity using Louvain algorithm.

    :param - graph: nx.Graph object representing butterfly similarity network
    return: dictionary mapping nodes to the communities they are assigned to
    """
    return louvain(graph, random_state=224)


def main():
    graph = utils.load_graph()
    position = utils.get_positions(graph)
    utils.make_dir('images/louvain')

    true_communities = utils.get_labels(graph, list(graph.nodes))
    utils.plot_communities(graph, position, true_communities, labels=True, title='Butterfly Similarity Network - True Communities', path='images/louvain/communities_true.png')

    communities = utils.group_communities(louvain_clustering(graph))
    utils.plot_communities(graph, position, communities, labels=False, title='Butterfly Similarity Network - Louvain Communities', path='images/louvain/communities_louvain.png')

    graph_nodes = sorted(list(graph.nodes))
    predictions = utils.predict_majority_class(graph, communities)
    preds = [predictions[n] for n in graph_nodes]
    labels = [graph.nodes[n]['label'] for n in graph_nodes]
    utils.report_classification_results(preds, labels, 'Confusion Matrix - Majority Label Predictions from Louvain Communities', 'images/louvain/cm_louvain.png')


if __name__=='__main__':
    main()
