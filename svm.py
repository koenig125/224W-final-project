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
    embedding_file_path = 'node2vec/embeddings/' + args.input
    X_train, X_test, y_train, y_test = utils.load_splits(embedding_file_path)
    X_train, X_test = utils.standardize_data(X_train, X_test)
    svc = SVC(gamma='scale')
    svc.fit(X_train, y_train)
    predictions = svc.predict(X_test)
    utils.make_dir('images/svc')
    cm_path = 'images/svc/cm_' + args.input[:-4] + '.png'
    utils.report_classification_results(predictions, y_test, 'Confusion Matrix - SVC', cm_path)


if __name__=='__main__':
	args = parse_args()
	main(args)
