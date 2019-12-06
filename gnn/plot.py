import argparse
import os
import sys

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

validation_dir = 'gnn/validation/'


def arg_parse():
    """
    Parses arguments that determine which files to plot.
    """
    parser = argparse.ArgumentParser(description='GNN arguments.')

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

    return parser.parse_args()


def plot_validation_accuracies(filenames, plot_name):
    validation_accuracies = []
    for f in filenames:
        v = np.load(validation_dir + f)
        validation_accuracies.append(v)
    
    epochs = [len(v) for v in validation_accuracies]
    assert (len(set(epochs)) == 1), 'Epochs not consistent'
    num_epochs = epochs[0]

    plt.clf()
    for i, f in enumerate(filenames):
        y = validation_accuracies[i]
        x = list(range(1, num_epochs + 1))
        c_index = int((i + 1) / len(filenames) * 255)
        c = cm.get_cmap('jet')(c_index)[:3]
        plt.plot(x, y, color=c, label=f)
        print('Best accuracy for', f, '-', max(y))
    
    plt.title('Validation Accuracy vs. Epochs for Node Classification')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    utils.make_dir('images/gnn')
    plt.savefig('images/gnn/' + plot_name)


def main(args):
    params = []
    if args.model_type is not None:
        params.append(args.model_type)
    if args.epochs is not None:
        params.append('_' + str(args.epochs) + '_')
    if args.num_layers is not None:
        params.append('_' + str(args.num_layers) + '_')
    if args.hidden_dim is not None:
        params.append('_' + str(args.hidden_dim) + '_')
    if args.dropout is not None:
        params.append(str(args.dropout))

    filenames = []
    for f in os.listdir('gnn/validation/'):
        if all(p in f for p in params):
            filenames.append(f)
    filenames.sort()
    save_path = '_'.join(params) + '.png'
    plot_validation_accuracies(filenames, save_path)


if __name__ == "__main__":
    args = arg_parse()
    main(args)
