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
    plt.tight_layout()
    utils.make_dir('images')
    plt.savefig(path)


def degree_distribution_by_species(graph, weighted):
    """
    Calculates the degree distribution for nodes in graph by species.

    :param - graph: nx.Graph object representing butterfly similarity network
    :param - weighted: boolean indicating whether to use weighted node degree
    return: dictionary mapping species to the average degree of nodes for that species.
    """
    species_to_avg_deg = {}
    counts = {}
    for n in graph.nodes:
        if weighted:
            d = int(graph.degree(n, 'weight'))
        else:
            d = graph.degree[n]
        s = graph.nodes[n]['label']
        if s in species_to_avg_deg:
            species_to_avg_deg[s] += d
            counts[s] += 1
        else:
            species_to_avg_deg[s] = d
            counts[s] = 1
    for s, c in counts.items():
        species_to_avg_deg[s] = species_to_avg_deg[s] / c
    lbls = np.array(list(species_to_avg_deg.keys()))
    degs = np.array(list(species_to_avg_deg.values()))
    idxs = np.argsort(lbls)
    lbls, degrees = lbls[idxs], degs[idxs]
    species = [utils.labels_to_species[l] for l in lbls]
    return (species, degrees)


def plot_degree_distribution_by_species(species, degrees, title, path):
    """
    Plot the average degree for each species.

    :param - species: list of species names
    :param - degrees: list of average degree for each species
    :param - title: title for the plot
    :param - path: save path for plot
    """
    plt.close('all')
    labels = range(len(species))
    plt.bar(labels, degrees)
    plt.xticks(labels, species, rotation=45, ha='right', fontsize=10)
    plt.xlabel('Species')
    plt.ylabel('Average Degree')
    plt.title(title)
    plt.tight_layout()
    utils.make_dir('images')
    plt.savefig(path)


def main():
    graph = utils.load_graph()

    x_w, y_w = degree_distribution(graph, weighted=True)
    x_u, y_u = degree_distribution(graph, weighted=False)
    plot_degree_distribution(x_w, y_w, 'Degree Distribution - Weighted', 'images/degdist_weighted.png')
    plot_degree_distribution(x_u, y_u, 'Degree Distribution - Unweighted', 'images/degdist_unweighted.png')

    s_w, d_w = degree_distribution_by_species(graph, weighted=True)
    s_u, d_u = degree_distribution_by_species(graph, weighted=False)
    plot_degree_distribution_by_species(s_w, d_w, 'Average Degree of Species - Weighted', 'images/avgdeg_weighted.png')
    plot_degree_distribution_by_species(s_u, d_u, 'Average Degree of Species - Unweighted', 'images/avgdeg_unweighted.png')


if __name__=='__main__':
    main()
