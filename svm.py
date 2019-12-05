"""
Script to run SVM classification on the node2vec embeddings 
of the BIOSNAP Leeds Butterfly Dataset with Network Enhancement.
"""

import argparse

import numpy as np
from sklearn.model_selection import GridSearchCV
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


def hyperparameter_search(svc, X, y):
    """
    Search hyperparameter space for best cross-validation score on data.

    return: Optimal model found from exhaustive search over parameter grid.
    """
    parameters = [
        {'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
        {'C': [0.1, 1, 10, 100], 'gamma': ['scale'], 'kernel': ['rbf', 'poly']},
    ]
    clf = GridSearchCV(svc, parameters, iid=False, cv=5)
    clf.fit(X, y)
    return clf


def main(args):
    # Load and standardize data.
    embedding_file_path = 'node2vec/embeddings/' + args.input
    X_train, X_test, y_train, y_test = utils.load_splits(embedding_file_path)
    X_train, X_test = utils.standardize_data(X_train, X_test)

    # Train classifier and make predictions.
    optimal_svc = hyperparameter_search(SVC(), X_train, y_train)
    predictions = optimal_svc.predict(X_test)

    # Report results.
    utils.make_dir('images/svc')
    cm_path = 'images/svc/cm_' + args.input[:-4] + '.png'
    utils.report_classification_results(predictions, y_test, 'Confusion Matrix - SVC', cm_path)


if __name__=='__main__':
	args = parse_args()
	main(args)
