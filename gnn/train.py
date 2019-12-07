"""
Trains a GNN on the BIOSNAP butterfly similarity dataset.
"""

import torch
from torch_geometric.data import DataLoader

import evaluate
import gnn
import gnn_utils
import models


def train(data, args):
    """
    Train the model specified by the parameters in args on the dataset given by data.

    :param - data: torch_geometric Data object holding node features, edges, and labels
    :param - args: user-specified arguments detailing GNN architecture and training
    return: DataLoader, trained model, and list of validation accuracies during training
    """
    num_classes = len(set([int(x) for x in data.y]))
    print('CUDA availability:', torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = DataLoader([data.to(device)], shuffle=True)
    model = models.GNN(data.num_node_features, args.hidden_dim, num_classes, args).to(device)
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
        
        val_acc = evaluate.eval(loader, model, is_test=False)
        if not len(validation_accuracies) or val_acc > max(validation_accuracies):
            # Save the model each time it achieves a new max val
            # accuracy. Previously saved models are overwritten.
            print('New max accuracy', val_acc, '- saving model...')
            gnn.save_model(args, model)
        validation_accuracies.append(val_acc)
        if epoch % 10 == 0:
            print('val:', val_acc)
            print('loss:', total_loss)
    
    return loader, model, validation_accuracies
