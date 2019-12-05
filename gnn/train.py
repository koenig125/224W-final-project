import argparse
import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid, TUDataset

import models
import gnn_utils

import os, sys
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


def train(dataset, args):
    test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = models.GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, args)
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


def test(loader, model, is_validation=False):
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
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    train(dataset, args) 


if __name__ == '__main__':
    main()
