"""
Loads the BIOSNAP butterfly similarity network in a pytorch geometric data format.
Reference: https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html.
"""

# pylint: disable=not-callable

import os
import sys

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


def get_labels(graph):
    """
    Retrieve the labels for all nodes in the graph.

    :param - graph: nx.Graph object representing the butterfly similarity network
    return: numpy array of labels, where index i gives the label for node with id i
    """
    node_ids = list(range(len(graph.nodes)))
    return np.array([graph.nodes[n]['label'] - 1 for n in node_ids])


def get_edges(graph):
    """
    Retrive edges and their weights from the graph.

    :param - graph: nx.Graph object representing the butterfly similarity network
    return: 2-tuple of torch tensors in the form (edge_index, edge_attr)
    """
    # Will have shape [2, 2 * num_edges] where row 1 represents source nodes and 
    # row 2 represents target nodes. There are 2 * num_edges because the edges are 
    # required in directed format but our graph is undirected, so we must count 
    # each edge twice, once for source --> target and once for target --> source. 
    edge_index = [[],[]]

    # Will have shape [num_edges, 1] and holds weights for each edge represented in edge_index.
    edge_attr = []
    
    edges = graph.edges.data('weight')
    for n1, n2, weight in edges:
        edge_index[0].extend([n1, n2])
        edge_attr.append([weight])
        edge_index[1].extend([n2, n1])
        edge_attr.append([weight])
    
    # Convert to tensors as expected by torch geometric data object
    edge_index = torch.tensor(np.array(edge_index), dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
    return edge_index, edge_attr


def get_masks(data):
    """
    Generate train, validation, and test masks and store them in the data object given.

    :param - data: torch_geometric Data object holding node features, edges, and labels
    """
    num_nodes = len(data.y)
    nids = utils.shuffle_ids(list(range(num_nodes)))

    # It is CRITICAL that test split specified below remains in unison with the
    # test split specified for the SVM models in svm.py in order to enable 
    # valid comparison of prediction results across the SVM and GNN models.
    val = int(num_nodes * .7)
    test = int(num_nodes * .8)

    train_ids = nids[:val]
    val_ids = nids[val:test]
    test_ids = nids[test:]

    data.train_mask = torch.tensor([1 if i in train_ids else 0 for i in range(num_nodes)], dtype=torch.bool)
    data.val_mask = torch.tensor([1 if i in val_ids else 0 for i in range(num_nodes)], dtype=torch.bool)
    data.test_mask = torch.tensor([1 if i in test_ids else 0 for i in range(num_nodes)], dtype=torch.bool)


def load_data(feature_type='identity', embedding_file=None):
    # Load graph.
    graph = utils.load_graph()
    node_ids = list(range(len(graph.nodes)))

    # Choose node features from identity, adjacency matrix, or embeddings.
    if feature_type == 'identity':
        node_features = np.eye(len(graph.nodes))
    elif feature_type == 'adjacency':
        node_features = nx.to_numpy_matrix(graph, node_ids)
    elif feature_type == 'embedding':
        embedding_path = 'node2vec/embeddings/' + embedding_file
        embeddings = utils.load_embeddings(embedding_path)
        node_features = np.array([embeddings[nid] for nid in node_ids])

    # Extract graph info to create torch geometric data object.
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(get_labels(graph), dtype=torch.long)
    edge_index, edge_attr = get_edges(graph)
    data = Data(x=x, edge_index=edge_index, y=y)

    # Obtain train/val/test splits.
    get_masks(data)

    return data
