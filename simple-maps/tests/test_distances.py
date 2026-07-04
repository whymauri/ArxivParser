import numpy as np
import pytest

from simple_maps.distances import METRICS, pairwise_distances


def naive_distance(x, y, metric):
    if metric == "sqeuclidean":
        return float(np.sum((x - y) ** 2))
    if metric == "euclidean":
        return float(np.sqrt(np.sum((x - y) ** 2)))
    if metric == "manhattan":
        return float(np.sum(np.abs(x - y)))
    if metric == "cosine":
        return float(1.0 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    raise AssertionError(metric)


@pytest.mark.parametrize("metric", METRICS)
def test_matches_naive_loop(metric):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(17, 5))
    Y = rng.normal(size=(11, 5))
    D = pairwise_distances(X, Y, metric=metric)
    assert D.shape == (17, 11)
    for i in range(17):
        for j in range(11):
            assert D[i, j] == pytest.approx(naive_distance(X[i], Y[j], metric), abs=1e-9)


@pytest.mark.parametrize("metric", METRICS)
def test_self_distances_are_zero(metric):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(9, 4))
    D = pairwise_distances(X, metric=metric)
    # The dot-product expansion leaves ~sqrt(eps) residue for euclidean.
    assert np.allclose(np.diag(D), 0.0, atol=1e-6)
    assert (D >= 0.0).all()


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        pairwise_distances(np.zeros((3, 2)), metric="minkowski-42")
