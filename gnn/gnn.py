"""
Walks through the steps of loading data, training the model, reporting results,
and saving learning and model information for the graph nueral network.
"""

import argparse
import os
import sys

import numpy as np
import torch

import data
import train
import gnn_utils

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
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')

    parser.set_defaults(model_type='GCN',
                        epochs=400,
                        num_layers=2,
                        hidden_dim=32,
                        dropout=0.0,
                        opt='adam',
                        opt_scheduler='none',
                        weight_decay=0.0,
                        lr=0.01)

    return parser.parse_args()


def save_info(args, model, validation_accuracies):
    file_name = '_'.join([args.model_type, str(args.epochs), str(args.num_layers), str(args.hidden_dim), str(args.dropout)])

    utils.make_dir('gnn/validation')
    np.save('gnn/validation/' + file_name + '.npy', validation_accuracies)

    utils.make_dir('gnn/trained_models')
    torch.save(model, 'gnn/trained_models/' + file_name + '.pt')


def main(args):
    data_tg = data.load_data()
    loader, model, validation_accuracies = train.train(data_tg, args)
    save_info(args, model, validation_accuracies)


if __name__ == '__main__':
    args = arg_parse()
    main(args)
