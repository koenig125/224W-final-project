"""
Script to calculate degree distribution for the BioSNAP butterfly similarity network.
"""

import matplotlib.pyplot as plt
import numpy as np

import utils


def degree_distribution(graph, weighted):
    """
    Calculates the degree distribution for nodes in graph.

    :param - graph: nx.Graph object representing butterfly similarity network
    :param - weighted: boolean indicating whether to use weighted node degree
    return: (list, list) tuple where the first list contains degree values and
    the second list contains the number of nodes with that degree.
    """
    degrees = {}
    for n in graph.nodes:
        if weighted:
            d = int(graph.degree(n, 'weight'))
        else:
            d = graph.degree[n]
        if d in degrees:
            degrees[d] += 1
        else:
            degrees[d] = 1
    degs = np.array(list(degrees.keys()))
    cnts = np.array(list(degrees.values()))
    idxs = np.argsort(degs)
    return (degs[idxs], cnts[idxs])


def plot_degree_distribution(degrees, counts, title, path):
    """
    Plot the degree distribution given by degrees and counts.

    :param - degrees: list of unique degree values
    :param - counts: list of counts for each degree value
    :param - title: title for the plot
    :param - path: save path for plot
    """
    plt.close('all')
    plt.plot(degrees, counts)
    plt.xlabel('Node Degree')
    plt.ylabel('Count')
    plt.title(title)
    utils.make_dir('images')
    plt.savefig(path)


def main():
    graph = utils.load_graph()
    x_w, y_w = degree_distribution(graph, weighted=True)
    x_u, y_u = degree_distribution(graph, weighted=False)
    plot_degree_distribution(x_w, y_w, 'Degree Distribution - Weighted', 'images/degdist_weighted.png')
    plot_degree_distribution(x_u, y_u, 'Degree Distribution - Unweighted', 'images/degdist_unweighted.png')


if __name__=='__main__':
    main()
