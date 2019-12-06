"""
Initializes the BIOSNAP butterfly similarity network in a data format
compatible with pytorch geometric. Then trains and evaluates a graph 
nueral network on the dataset, reporting learning behavior and results.
"""

# pylint: disable=no-member
# pylint: disable=not-callable

import argparse
import os
import sys

import networkx as nx
import numpy as np
import torch
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, DataLoader

import gnn_utils
import models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


def arg_parse():
    """
    Parses the arguments for the GNN architecture and training.
    """
    parser = argparse.ArgumentParser(description='GNN arguments.')
    gnn_utils.parse_optimizer(parser)

    parser.add_argument('--model_type', type=str,
                        help='Type of GNN model.')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')

    parser.set_defaults(model_type='GCN',
                        epochs=400,
                        batch_size=1,
                        num_layers=2,
                        hidden_dim=32,
                        dropout=0.0,
                        opt='adam',
                        opt_scheduler='none',
                        weight_decay=0.0,
                        lr=0.01)

    return parser.parse_args()


def get_labels(graph):
    """
    Retrieve the labels for all nodes in the graph provided.

    :param - graph: nx.Graph object representing the butterfly similarity network
    return: numpy array of labels, where index i gives the label for node with id i
    """
    node_ids = list(range(len(graph.nodes)))
    return np.array([graph.nodes[n]['label'] for n in node_ids])


def get_edges(graph):
    """
    Retrive edges and their weights from the graph provided.

    :param - graph: nx.Graph object representing the butterfly similarity network
    return: 2-tuple of torch tensors in the form (edge_index, edge_attr), where
    edge_index and edge_attr fulfill the specifications for the torch geometric
    'Data' object (https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html)
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


def train(data, args):
    """
    Train the model specified by tge parameters in args on the dataset given by data.

    :param - data: torch_geometric Data object holding node features, edges, and labels
    :param - args: user-specified arguments detailing GNN architecture and training
    return: DataLoader, trained model, and list of validation accuracies during training
    """
    num_classes = len(set(data.y))
    loader = DataLoader([data], batch_size=args.batch_size, shuffle=True)
    model = models.GNN(data.num_node_features, args.hidden_dim, num_classes, args)
    scheduler, optimizer = gnn_utils.build_optimizer(args, model.parameters())

    validation_accuracies = []
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch)[batch.train_mask]
            label = batch.y[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_acc = evaluate(loader, model, is_test=False)
        validation_accuracies.append(val_acc)
        if epoch % 10 == 0:
            print('val:', val_acc)
            print('loss:', total_loss)
    
    return loader, model, validation_accuracies


def evaluate(loader, model, is_test=False):
    """
    Evaluate model performance on data object (graph) in loader.

    :param - loader: torch_geometric DataLoader for BIOSNAP dataset
    :param - model: trained GNN model ready for making predictions
    :param - is_test: boolean indicating to evaluate on test or eval split
    """
    model.eval()
    data = [data for data in loader][0]
    mask = data.test_mask if is_test else data.val_mask
    with torch.no_grad():
        pred = model(data).max(dim=1)[1][mask]
        label = data.y[mask]
    correct = pred.eq(label).sum().item()
    total = mask.sum().item()
    return correct / total


def main(args):
    # Load graph.
    graph = utils.load_graph()

    # Extract graph info to create torch geometric data object.
    x = torch.tensor(np.eye(len(graph.nodes)), dtype=torch.float)
    y = torch.tensor(get_labels(graph), dtype=torch.long)
    edge_index, edge_attr = get_edges(graph)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # Obtain train/val/test splits.
    get_masks(data)

    # Train model.
    loader, model, validation_accuracies = train(data, args)
    print('Best accuracy:', max(validation_accuracies))

    # Save validation accuracies from learning.
    utils.make_dir('gnn/val_accuracies')
    np.save('gnn/val_accuracies/' + args.model_type, validation_accuracies)


if __name__ == '__main__':
    args = arg_parse()
    main(args)
