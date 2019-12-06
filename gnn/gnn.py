"""
Walks through the steps of loading data, training the model, reporting results,
and saving learning and model information for the graph nueral network.
"""

import argparse
import os
import sys

import numpy as np

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


def main(args):
    # Load data and train model.
    data_tg = data.load_data()
    loader, model, validation_accuracies = train.train(data_tg, args)
    print('Best accuracy:', max(validation_accuracies))

    # Save validation accuracies.
    utils.make_dir('gnn/val_accuracies')
    np.save('gnn/val_accuracies/' + args.model_type, validation_accuracies)


if __name__ == '__main__':
    args = arg_parse()
    main(args)
