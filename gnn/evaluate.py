"""
Evaluates a GNN on the BIOSNAP butterfly similarity dataset.
"""

import argparse

import torch
from torch_geometric.data import DataLoader

import data
import gnn_utils


def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')

    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Filename of pytorch model to load.')
    parser.add_argument('-t', '--test', type=bool, default=False,
                        help='Evaluate on test split.')
                        
    return parser.parse_args()


def eval(loader, model, is_test=False):
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


def main(args):
    f = gnn_utils.models_dir + args.model
    model = torch.load(f)
    data_tg = data.load_data()
    loader = DataLoader([data_tg], shuffle=True)
    print(eval(loader, model, is_test=args.test))


if __name__ == '__main__':
    args = arg_parse()
    main(args)
