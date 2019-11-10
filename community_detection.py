"""
Runs Louvain community detection algorithm on the Leeds Butterfly dataset
and provides species distribution analysis for detected communities.
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


def community_info(graph, communities):
    """
    Provide basic information about communities provided.

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
    species = get_species(graph, nodes)
    for s, node_ids in species.items():
        count = len(node_ids)
        percent = count / len(nodes)
        print(template.format(s, count, '{0:.3f}'.format(percent)))


def get_species(graph, nodes):
    """
    Get the species of each node and group nodes by species.

    :param - graph: nx.Graph object representing butterfly similarity network
    :param - nodes: list of node ids from graph
    return: dictionary mapping species to nodes
    """
    species = {}
    for node in nodes:
        l = graph.nodes[node]['label']
        s = labels_to_species[l]
        if s in species:
            species[s].append(node)
        else:
            species[s] = [node]
    return species


def main():
    graph = load_graph()
    communities = partition(graph)
    community_info(graph, communities)


if __name__=='__main__':
    main()
