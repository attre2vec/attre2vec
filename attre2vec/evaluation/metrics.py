"""Functions for computing edge clustering."""
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(
    embs: np.ndarray, y_true: np.ndarray, n_classes: Optional[int] = None
) -> float:
    """Calculates clustering accuracy.

    Args:
        embs: embeddings to be clustered, numpy.array with shape
            `(n_samples, n_features)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
        n_classes: number of classes, if `None` is provided then number of
            classes is taken as `np.max(y_true) + 1`
    Returns:
        accuracy, in [0,1]

    """
    if n_classes is None:
        n_clusters = np.max(y_true) + 1
    else:
        n_clusters = n_classes
    clusterer = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0)
    y_pred = clusterer.fit_predict(embs)

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(np.max(w) - w)
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size


def _test_acc():
    rng = np.random.RandomState(0)
    embs = rng.normal(0, 1, size=(1000, 32))
    y_true = rng.randint(0, 4, size=(1000,))
    print("Acc: {}".format(acc(embs, y_true)))  # should output Acc: 0.276


if __name__ == "__main__":
    _test_acc()
