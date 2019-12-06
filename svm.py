"""
Script to run SVM classification on the node2vec embeddings 
of the BIOSNAP Leeds Butterfly Dataset with Network Enhancement.
"""

import argparse

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
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


def load_splits(embeddings_path):
    """
    Generate train/test splits for the butterfly embeddings. Embeddings 
    need to be generated prior by running node2vec as specified here: 
    https://github.com/koenig125/224W-final-project.

    return: 4-tuple holding the training embeddings, training labels, testing
    embeddings, and testing labels for the BIOSNAP butterfly similarity network.
    """
    labels = utils.load_labels()
    embeddings = utils.load_embeddings(embeddings_path)
    num_nodes = len(labels)

    nids = utils.shuffle_ids(list(range(num_nodes)))
    X = np.array([embeddings[n] for n in nids])
    y = np.array([labels[n] for n in nids])

    test = int(num_nodes * .8)
    X_train, y_train = X[:test], y[:test]
    X_test, y_test = X[test:], y[test:]
    return X_train, X_test, y_train, y_test


def standardize_data(X_train, X_test):
    """
    Standardizes X_train to zero mean and unit variance, then applies the 
    transformation that was executed on X_train to X_test.
    """
    scaler = StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    return X_train_transformed, X_test_transformed


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
    X_train, X_test, y_train, y_test = load_splits(embedding_file_path)
    X_train, X_test = standardize_data(X_train, X_test)

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
