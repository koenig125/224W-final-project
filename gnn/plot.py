import os
import sys

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

validation_dir = 'gnn/validation/'


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


def main():
    model_types = ['GCN', 'GraphSage', 'GAT']
    epochs = ['400']
    num_layers = ['2', '3', '4']
    hidden_dim = ['32', '64', '128']
    dropout = ['0.0', '0.1']

    for m in model_types:
        for e in epochs:
            for n in num_layers:
                filenames = []
                for h in hidden_dim:
                    for d in dropout:
                        f = '_'.join([m, e, n, h, d]) + '.npy'
                        filenames.append(f)
                plot_validation_accuracies(filenames, 'm=' + m + '_n=' + n + '.png')


if __name__ == "__main__":
    main()
