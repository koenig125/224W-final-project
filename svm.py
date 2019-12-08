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


def load_labels():
    """
    Load butterfly similarity network labels.

    return: dictionary mapping node ids to labels (1-10)
    """
    labels = {}
    nf = open(utils.nodefile, 'r')
    for line in nf:
        if line[0] == '#': continue # skip comments
        node_id, label = line[:-1].split('\t')
        node_id, label = int(node_id), int(label)
        labels[node_id] = label
    return labels


def load_splits(embeddings_path):
    """
    Generate train/test splits for the butterfly embeddings. Embeddings 
    need to be generated prior by running node2vec as specified here: 
    https://github.com/koenig125/224W-final-project.

    return: 4-tuple holding the training data, training labels, testing
    data, and testing labels for the BIOSNAP butterfly similarity network.
    """
    labels = load_labels()
    embeddings = utils.load_embeddings(embeddings_path)
    num_nodes = len(labels)

    nids = utils.shuffle_ids(list(range(num_nodes)))
    X = np.array([embeddings[n] for n in nids])
    y = np.array([labels[n] for n in nids])

    # It is CRITICAL that test split specified below remains in unison with the
    # test split specified for the GNN models in train.py in order to enable 
    # valid comparison of prediction results across the SVM and GNN models.
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
    print('Optimal parameters:', optimal_svc.best_params_)
    predictions = optimal_svc.predict(X_test)

    # Report results.
    utils.make_dir('images/svc')
    cm_path = 'images/svc/cm_' + args.input[:-4] + '.png'
    utils.accuracy(predictions, y_test)
    utils.confusion_matrix(predictions, y_test, 'Confusion Matrix - SVC', cm_path)


if __name__=='__main__':
	args = parse_args()
	main(args)
