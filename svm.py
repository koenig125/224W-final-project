"""
Script to run SVM classification on the node2vec embeddings 
of the BIOSNAP Leeds Butterfly Dataset with Network Enhancement.
"""

import argparse

import numpy as np
from sklearn.svm import SVC

import utils


def parse_args():
    """
    Parses the svm arguments.
    """
    parser = argparse.ArgumentParser(description="Run SVM.")

    parser.add_argument('--input', nargs='?', default='emb_p=1_q=1.emb', 
                        help='Name of input file for node embeddings')

    return parser.parse_args()


def main(args):
    # Read in node embeddings and labels
    embedding_file_path = 'node2vec/embeddings/' + args.input
    embeddings = utils.load_embeddings(embedding_file_path)
    labels = utils.load_labels()
    num_nodes = len(labels)

    # Shuffle embeddings and labels
    node_ids = list(range(num_nodes))
    np.random.shuffle(node_ids)
    X = np.array([embeddings[n] for n in node_ids])
    y = np.array([labels[n] for n in node_ids])

    # Split data into train/dev/test
    test = int(num_nodes * .8)
    X_train, y_train = X[:test], y[:test]
    X_test, y_test = X[test:], y[test:]

    # Train and evaluate SVC
    svc = SVC(gamma='auto')
    svc.fit(X_train, y_train)
    predictions = svc.predict(X_test)
    utils.make_dir('images/svc')
    cm_path = 'images/svc/cm_' + args.input[:-4] + '.png'
    utils.report_classification_results(predictions, y_test, 'Confusion Matrix - SVC', cm_path)


if __name__=='__main__':
	args = parse_args()
	main(args)
