# pylint: disable=no-member
# pylint: disable=not-callable

import argparse
import os
import sys
import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.utils import from_networkx

import gnn_utils
import models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    gnn_utils.parse_optimizer(parser)

    parser.add_argument('--model_type', type=str,
                        help='Type of GNN model.')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')

    parser.set_defaults(model_type='GCN',
                        batch_size=32,
                        epochs=50,
                        num_layers=2,
                        hidden_dim=32,
                        dropout=0.0,
                        dataset='cora',
                        opt='adam',
                        opt_scheduler='none',
                        weight_decay=0.0,
                        lr=0.01)

    return parser.parse_args()


def load_labels(graph):
    node_ids = list(range(len(graph.nodes)))
    return np.array([graph.nodes[n]['label'] for n in node_ids])


def load_edges(graph):
    edge_attr = []
    edge_index = [[],[]]
    edges = graph.edges.data('weight')
    for n1, n2, weight in edges:
        edge_index[0].extend([n1, n2])
        edge_index[1].extend([n2, n1])
        edge_attr.append([weight])
        edge_attr.append([weight])
    return np.array(edge_index), np.array(edge_attr)


def load_masks(data):
    num_nodes = len(data.y)
    nids = utils.shuffle_ids(list(range(num_nodes)))

    val_threshold = int(num_nodes * .6)
    test_threshold = int(num_nodes * .8)

    train_ids = nids[:val_threshold]
    val_ids = nids[val_threshold:test_threshold]
    test_ids = nids[test_threshold:]

    data.train_mask = torch.tensor([1 if i in train_ids else 0 for i in range(num_nodes)], dtype=torch.bool)
    data.val_mask = torch.tensor([1 if i in val_ids else 0 for i in range(num_nodes)], dtype=torch.bool)
    data.test_mask = torch.tensor([1 if i in test_ids else 0 for i in range(num_nodes)], dtype=torch.bool)


def train(data, args):
    loader = DataLoader([data], batch_size=args.batch_size, shuffle=True)
    num_classes = len(set(data.y))
    model = models.GNNStack(data.num_node_features, args.hidden_dim, num_classes, args)
    scheduler, opt = gnn_utils.build_optimizer(args, model.parameters())

    validation_accuracies = []
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        print(total_loss)
        validation_accuracies.append(test(loader, model, is_validation=True))

        if epoch % 10 == 0:
            test_acc = test(loader, model)
            print(test_acc, '  test')
    utils.make_dir('gnn/val_accuracies')
    np.save('gnn/val_accuracies/' + args.model_type, validation_accuracies)


def test(loader, model, is_validation=True):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
            label = data.y
        mask = data.val_mask if is_validation else data.test_mask
        pred = pred[mask]
        label = data.y[mask]
        correct += pred.eq(label).sum().item()
    
    total = 0
    for data in loader.dataset:
        total += torch.sum(data.val_mask if is_validation else data.test_mask).item()
    return correct / total


def main():
    args = arg_parse()
    graph = utils.load_graph()

    x = torch.tensor(np.eye(len(graph.nodes)), dtype=torch.float)
    y = torch.tensor(load_labels(graph), dtype=torch.long)

    edge_index, edge_attr = load_edges(graph)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    load_masks(data)
    train(data, args)


if __name__ == '__main__':
    main()
