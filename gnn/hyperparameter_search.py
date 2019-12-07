"""
Explores hyperparameter space for GNN training.
"""

import argparse
import copy
import os
import sys

import data
import gnn_utils
from gnn import train_and_save_gnn


def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    gnn_utils.parse_optimizer(parser)

    parser.add_argument('-nf', '--node_features', nargs='+', type=str, required=True,
                        help='Types of node features.')
    parser.add_argument('-ef', '--embedding_file', type=str, default=None,
                        help='File with node2vec embeddings.')
    parser.add_argument('-m', '--model_type', nargs='+', type=str, required=True,
                        help='Types of GNN models.')
    parser.add_argument('-e', '--epochs', nargs='+', type=int, required=True,
                        help='Number of training epochs')
    parser.add_argument('-nl', '--num_layers', nargs='+', type=int, required=True,
                        help='Number of graph conv layers')
    parser.add_argument('-hd', '--hidden_dim', nargs='+', type=int, required=True,
                        help='Training hidden size')
    parser.add_argument('-d', '--dropout', nargs='+', type=float, required=True,
                        help='Dropout rate')
    
    parser.set_defaults(opt='adam',
                        opt_scheduler='none',
                        weight_decay=0.0,
                        lr=0.01)

    return parser.parse_args()


def hyperparameter_search(data, args):
    for f in args.node_features:
        for e in args.epochs:
            for m in args.model_type:
                for n in args.num_layers:
                    for h in args.hidden_dim:
                        for d in args.dropout:
                            ef = args.embedding_file
                            print('Training model with params:', [f, ef, e, m, n, h, d])
                            new_args = copy.deepcopy(args)
                            new_args.node_features = f
                            new_args.embedding_file = ef
                            new_args.epochs = e
                            new_args.model_type = m
                            new_args.num_layers = n
                            new_args.hidden_dim = h
                            new_args.dropout = d
                            train_and_save_gnn(data, new_args)


def main(args):
    data_tg = data.load_data()
    hyperparameter_search(data_tg, args)


if __name__ == '__main__':
    args = arg_parse()
    main(args)
