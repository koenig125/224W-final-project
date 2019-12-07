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

    parser.add_argument('-nf', '--node_features', type=str, required=True,
                        help='Type of node features.')
    parser.add_argument('-ef', '--embedding_file', type=str, default=None,
                        help='File with node2vec embeddings.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Filename of pytorch model to load.')
    parser.add_argument('-t', '--test', default=False, action="store_true",
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


def eval_saved_model(filename, node_features, embedding_file, is_test):
    f = gnn_utils.models_dir + filename
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(f, map_location=torch.device(device))
    data_tg = data.load_data(node_features, embedding_file)
    loader = DataLoader([data_tg.to(device)], shuffle=True)
    print(eval(loader, model, is_test))


def main(args):
    eval_saved_model(args.model, args.node_features, args.embedding_file, args.test)


if __name__ == '__main__':
    args = arg_parse()
    main(args)
