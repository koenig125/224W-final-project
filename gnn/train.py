"""
Trains and evaluates a GNN on the BIOSNAP butterfly similarity dataset.
"""

# pylint: disable=no-member
# pylint: disable=not-callable

import torch
from torch_geometric.data import DataLoader

import data
import gnn_utils
import models


def train(data, args):
    """
    Train the model specified by the parameters in args on the dataset given by data.

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
    :param - is_test: boolean indicating whether to evaluate on test or eval split
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
