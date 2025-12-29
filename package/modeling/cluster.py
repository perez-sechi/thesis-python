import copy
import numpy as np
import networkx as nx
from itertools import permutations
from random import randint, sample


def communities_from_index(community_index):
    """
    Convert a list of cluster indices to a list of clusters.
    """
    clusters_dict = {}
    for idx, cluster in enumerate(community_index):
        if cluster not in clusters_dict or clusters_dict[cluster] is None:
            clusters_dict[cluster] = []
        clusters_dict[cluster].append(idx)

    return list(clusters_dict.values())


def modularity(i, cj, community_index, m, K, Kc, R_array):
    """
    Calculate the modularity of moving node i to community cj.
    https://en.wikipedia.org/wiki/Louvain_method
    Parameters:
    i (int): The index of the node to move.
    cj (int): The index of the community to move the node to.
    community_index (list[int]): The current community assignment of each node.
    m (int): The sum of all edge weights in the graph.
    K (list[int]): The sum of edge weights for each node.
    Kc (list[int]): The sum of edge weights for each community.
    R_array (np.array): The modularity matrix of the graph.
    Returns:
    float: The modularity of moving node i to community cj.
    """
    members_cj = [
        idx for idx, c in enumerate(community_index)
        if c == cj
    ]
    ki_in = 2 * np.sum(R_array[i, members_cj])
    return (ki_in / m) - (2 * K[i] * Kc[cj]) / (m ** 2)


def additional_louvain(A, R):
    """
    Perform an additional Louvain clustering on the given adjacency matrix.
    Parameters:
    A (list[list[int]]): Adjacency matrix representing the graph.
    R (list[list[float]]): Matrix used to create a directed graph for 
    modularity calculation.
    Returns:
    list[int]: The partition of the nodes after applying the Louvain method.
    """
    n = len(A)
    o = sample(range(n), n)
    community_index = list(range(n))

    R_array = np.array(R)
    R_array = R_array + R_array.transpose()
    
    m = np.sum(R_array)
    K = [sum(row) for row in R_array]
    Kc = [sum(row) for row in R_array]
    
    improved = True
    iteration = 0
    while improved:
        iteration += 1
        improved = False
        for i in o:
            ci = community_index[i]
            neighbors = np.nonzero(A[i])[0].tolist()
            new_community = ci
            community_index[i] = -1

            Kc[ci] = Kc[ci] - K[i]

            previous_modularity = modularity(
                i, ci, community_index, m, K, Kc, R_array
            )
            best_increase = 0
            temporal_new_community = None
            for j in neighbors:
                if i == j:
                    continue

                cj = community_index[j]
                current_modularity = modularity(
                    i, cj, community_index, m, K, Kc, R_array
                )
                increase = current_modularity - previous_modularity

                if increase > best_increase:
                    best_increase = increase
                    temporal_new_community = cj

            if best_increase > 0:
                new_community = temporal_new_community

            community_index[i] = new_community
            Kc[new_community] = Kc[new_community] + K[i]

            if new_community != ci:
                improved = True

    return communities_from_index(community_index)


def duo_louvain(A, R):
    """
    Perform a clustering operation using a modified Louvain method.
    This function iteratively applies the Louvain method to cluster the nodes
    of a graph represented by adjacency matrix A and modularity matrix R.
    Parameters:
    A (list[list[int]]): The adjacency matrix of the graph.
    R (list[list[float]]): The modularity matrix of the graph.
    Returns:
    list[int]: A list of sets, where each set contains the indices of nodes
    in the same cluster.
    """
    n = len(A)
    P1 = []
    P2 = [[i] for i in range(n)]
    A1 = np.copy(A)
    A2 = np.copy(A)
    R1 = np.copy(R)
    R2 = np.copy(R)

    is_shape_consistent = (
        len(A2) > 1
        and A2.shape[0] == A2.shape[1]
        and R2.shape[0] == R2.shape[1]
        and A2.shape == R2.shape
    )

    iteration = 0
    partitions = {
        iteration: copy.deepcopy(P2),
    }

    while not np.array_equal(P1, P2) and is_shape_consistent:
        iteration += 1
        P1 = P2
        P2 = additional_louvain(A2.tolist(), R2.tolist())
        partitions[iteration] = copy.deepcopy(P2)

        A2 = np.zeros((len(P2), len(P2)))
        R2 = np.zeros((len(P2), len(P2)))

        for cdx, c in enumerate(P2):
            for i in c:
                for ddx, d in enumerate(P2):
                    for j in d:
                        A2[cdx, ddx] = 1 \
                            if A2[cdx, ddx] > 0 or A1[i, j] > 0 \
                            else 0
                        R2[cdx, ddx] += R1[i, j]

        A1 = np.copy(A2)
        R1 = np.copy(R2)

        is_shape_consistent = (
            len(A2) > 1
            and A2.shape[0] == A2.shape[1]
            and R2.shape[0] == R2.shape[1]
            and A2.shape == R2.shape
        )

    communities = partitions[iteration]
    while iteration > 0:
        iteration -= 1
        for cdx, c in enumerate(communities):
            new_community = []
            for d in c:
                new_community.extend(partitions[iteration][d])
            communities[cdx] = new_community

    return communities
