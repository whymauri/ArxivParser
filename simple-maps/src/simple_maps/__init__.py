"""simple-maps: simple and effective manifold approximators, few dependencies.

Public API:
    UMAP, TriMap, PaCMAP  -- sklearn-style estimators (fit_transform/transform)
    load                  -- restore a saved estimator from .npz
    knn, pairwise_distances -- core ops, reusable on their own
"""

from ._base import load
from .classify import KNNClassifier
from .distances import METRICS, pairwise_distances
from .neighbors import knn, knn_query
from .pacmap_ import PaCMAP
from .parametric import ParametricPaCMAP, ParametricTriMap, ParametricUMAP
from .trimap_ import TriMap
from .umap_ import UMAP

__version__ = "0.3.0"

_ESTIMATORS = {
    cls.__name__: cls
    for cls in (
        UMAP,
        TriMap,
        PaCMAP,
        ParametricUMAP,
        ParametricTriMap,
        ParametricPaCMAP,
        KNNClassifier,
    )
}

__all__ = [
    "UMAP",
    "TriMap",
    "PaCMAP",
    "ParametricUMAP",
    "ParametricTriMap",
    "ParametricPaCMAP",
    "KNNClassifier",
    "load",
    "knn",
    "knn_query",
    "pairwise_distances",
    "METRICS",
    "__version__",
]
