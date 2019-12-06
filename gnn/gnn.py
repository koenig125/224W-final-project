"""
Walks through the steps of loading data, training a model, reporting results,
and saving learning and model information for a single graph nueral network.
"""

import argparse
import os
import sys

import numpy as np
import torch

import data
import evaluate
import gnn_utils
import train

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    gnn_utils.parse_optimizer(parser)

    parser.add_argument('-m', '--model_type', type=str,
                        help='Type of GNN model.')
    parser.add_argument('-e', '--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('-nl', '--num_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('-hd', '--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('-d', '--dropout', type=float,
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


def save_model(args, model):
    file_name = '_'.join([args.model_type, str(args.epochs), str(args.num_layers), str(args.hidden_dim), str(args.dropout)])
    utils.make_dir(gnn_utils.models_dir)
    torch.save(model, gnn_utils.models_dir + file_name + '.pt')


def save_accuracies(args, validation_accuracies):
    file_name = '_'.join([args.model_type, str(args.epochs), str(args.num_layers), str(args.hidden_dim), str(args.dropout)])
    utils.make_dir(gnn_utils.validation_dir)
    np.save(gnn_utils.validation_dir + file_name + '.npy', validation_accuracies)


def train_and_save_gnn(data, args):
    # Model is saved during training, so no need to save again here
    loader, model, validation_accuracies = train.train(data, args)
    save_accuracies(args, validation_accuracies)
    return model


def main():
    args = arg_parse()
    data_tg = data.load_data()
    train_and_save_gnn(data_tg, args)


if __name__ == '__main__':
    main()
