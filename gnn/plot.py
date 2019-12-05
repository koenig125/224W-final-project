import numpy as np
from matplotlib import pyplot as plt

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


def main():
    epochs = 50

    vacc_GCN = np.load('gnn/val_accuracies/GCN.npy')
    vacc_SAGE = np.load('gnn/val_accuracies/GraphSage.npy')
    vacc_GAT = np.load('gnn/val_accuracies/GAT.npy')

    plt.clf()
    plt.plot(list(range(1, epochs + 1)), vacc_GCN, color = 'r', label = 'GCN')
    plt.plot(list(range(1, epochs + 1)), vacc_SAGE, linestyle = 'dashed', color = 'g', label = 'SAGE')
    plt.plot(list(range(1, epochs + 1)), vacc_GAT, linestyle = 'dotted', color = 'b', label = 'GAT')
    plt.title('Validation Accuracy vs. Epochs for Node Classification')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    utils.make_dir('images/gnn')
    plt.savefig('images/gnn/node-classification-training.png')


if __name__ == '__main__':
	main()
