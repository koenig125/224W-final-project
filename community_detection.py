"""
Runs Louvain community detection algorithm on the Leeds Butterfly dataset
and provides species distribution analysis for detected communities. Also
provides classification accuracy based on majority-label predictions from
the communities identified by Louvain.
"""

import networkx as nx
from community import best_partition as louvain

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


def report_classification_accuracy(preds, labels):
    """
    Calculate the classification accuracy given predictions and labels.

    :param - preds: dictionary mapping node ids to predicted labels
    :param - labels: dictionary mapping node ids to ground-truth labels
    """
    all_nodes = labels.keys()
    correct = [preds[n] == labels[n] for n in all_nodes]
    accuracy = sum(correct) / len(all_nodes)
    print('Majority Label Classification Accuracy: {0:.5f}'.format(accuracy))


def main():
    graph = load_graph()
    communities = partition(graph)
    report_species_distribution(graph, communities)

    predictions = predict(graph, communities)
    ground_truth = {n: graph.nodes[n]['label'] for n in list(graph.nodes)}
    report_classification_accuracy(predictions, ground_truth)


if __name__=='__main__':
    main()
