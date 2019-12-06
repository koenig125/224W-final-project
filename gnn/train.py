"""
Trains a GNN on the BIOSNAP butterfly similarity dataset.
"""

from torch_geometric.data import DataLoader

import gnn_utils
import models
import evaluate


def train(data, args):
    """
    Train the model specified by the parameters in args on the dataset given by data.

    :param - data: torch_geometric Data object holding node features, edges, and labels
    :param - args: user-specified arguments detailing GNN architecture and training
    return: DataLoader, trained model, and list of validation accuracies during training
    """
    num_classes = len(set(data.y))
    loader = DataLoader([data], shuffle=True)
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
        val_acc = evaluate.eval(loader, model, is_test=False)
        validation_accuracies.append(val_acc)
        if epoch % 10 == 0:
            print('val:', val_acc)
            print('loss:', total_loss)
    
    return loader, model, validation_accuracies
