"""Functions for sampling negative and positive neighbor edges."""
import numpy as np


def _get_neighborhood(xe, rws, graph):
    neighborhood = {}
    for edge in xe:
        neighbors = list(set(
            e
            for walk in rws[edge].values()
            for edges in walk
            for e in edges
            if e in graph.edges()
        ))
        neighborhood[tuple(edge)] = neighbors
    return neighborhood


def _sample_neighbors(xe, nh, num_samples):
    samples = []
    xe_idxs = []

    for idx, edge in enumerate(xe):
        neighbors = np.array(nh[edge])

        if neighbors.shape[0] == 0:
            continue

        sample_idxs = np.random.choice(
            a=neighbors.shape[0],
            size=num_samples,
            replace=neighbors.shape[0] < num_samples,
        )
        samples.extend(neighbors[sample_idxs])
        xe_idxs.extend([idx] * num_samples)

    samples, xe_idxs = np.array(samples), np.array(xe_idxs)
    return samples, xe_idxs


def _sample_negatives(xe, nh, graph, num_samples):
    samples = []
    xe_idxs = []

    all_edges = list(graph.edges())

    for idx, edge in enumerate(xe):
        enh = set(nh[edge])
        enh.add(edge)

        i = 0
        while i < num_samples:
            sample_idx = np.random.randint(low=0, high=len(all_edges))
            if all_edges[sample_idx] in enh:
                continue
            samples.append(all_edges[sample_idx])
            xe_idxs.append(idx)
            i += 1

    samples, xe_idxs = np.array(samples), np.array(xe_idxs)
    return samples, xe_idxs


def precompute_samples(xe, rws, graph, num_pos, num_neg):
    """Computes positive and negative samples for given edges."""
    nh = _get_neighborhood(xe=xe, rws=rws, graph=graph)

    xe_plus, idxs_plus = _sample_neighbors(
        xe=xe, nh=nh, num_samples=num_pos,
    )
    xe_minus, idxs_minus = _sample_negatives(
        xe=xe, nh=nh, graph=graph, num_samples=num_neg,
    )

    return {
        'plus': {'xe': xe_plus, 'idxs': idxs_plus},
        'minus': {'xe': xe_minus, 'idxs': idxs_minus},
    }
