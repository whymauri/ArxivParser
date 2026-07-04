"""Benchmark simple-maps against reference implementations (if installed).

Usage:
    python benchmarks/run_benchmarks.py [--n 5000] [--dim 50] [--k 10]

Reference implementations (umap-learn, trimap, pacmap) are optional; any
that fail to import are skipped. Reported metrics:

- fit     : wall-clock fit_transform on n points
- infer   : wall-clock transform of n//10 held-out points (where supported)
- knn@k   : fraction of high-dim k-neighborhoods preserved in the embedding
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from simple_maps import (
    PaCMAP,
    ParametricPaCMAP,
    ParametricTriMap,
    ParametricUMAP,
    TriMap,
    UMAP,
)
from simple_maps.datasets import gaussian_blobs, swiss_roll
from simple_maps.metrics import knn_preservation


def bench_one(name, fit_transform, transform, X_train, X_test, k):
    t0 = time.perf_counter()
    Y = fit_transform(X_train)
    fit_s = time.perf_counter() - t0

    infer_s = None
    if transform is not None:
        t0 = time.perf_counter()
        transform(X_test)
        infer_s = time.perf_counter() - t0

    score = knn_preservation(X_train, Y, n_neighbors=k)
    infer_str = f"{infer_s:8.3f}s" if infer_s is not None else "       --"
    print(f"  {name:<24} fit {fit_s:8.2f}s   infer {infer_str}   knn@{k} {score:.3f}")


def candidates(X_train):
    """Yield (name, factory, make_transform) for ours + references.

    make_transform(model) returns a callable over new points, or None when
    the implementation has no out-of-sample support.
    """
    plain = lambda model: model.transform
    yield "simple-maps UMAP", lambda: UMAP(random_state=0), plain
    yield "simple-maps TriMap", lambda: TriMap(random_state=0), plain
    yield "simple-maps PaCMAP", lambda: PaCMAP(random_state=0), plain
    yield "simple-maps ParamUMAP", lambda: ParametricUMAP(random_state=0), plain
    yield "simple-maps ParamTriMap", lambda: ParametricTriMap(random_state=0), plain
    yield "simple-maps ParamPaCMAP", lambda: ParametricPaCMAP(random_state=0), plain

    try:
        import umap as umap_learn

        yield "umap-learn", lambda: umap_learn.UMAP(random_state=0), plain
    except ImportError:
        print("  (umap-learn not installed; skipping)")
    try:
        import trimap as trimap_ref

        yield "trimap (ref)", lambda: trimap_ref.TRIMAP(), lambda model: None
    except ImportError:
        print("  (trimap not installed; skipping)")
    try:
        import pacmap as pacmap_ref

        # pacmap's transform needs the training set back as the basis.
        yield (
            "pacmap (ref)",
            lambda: pacmap_ref.PaCMAP(random_state=0),
            lambda model: (lambda X_new: model.transform(X_new, basis=X_train)),
        )
    except ImportError:
        print("  (pacmap not installed; skipping)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--dim", type=int, default=50)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    datasets = {
        f"blobs(n={args.n}, d={args.dim})": gaussian_blobs(
            n_samples=args.n, n_features=args.dim, n_clusters=8, random_state=0
        )[0],
        f"swiss_roll(n={args.n})": swiss_roll(n_samples=args.n, random_state=0)[0],
    }

    for ds_name, X in datasets.items():
        print(f"\n{ds_name}")
        rng = np.random.default_rng(1)
        X_test = X[rng.choice(len(X), size=max(len(X) // 10, 1), replace=False)]
        for name, factory, make_transform in candidates(X):
            model = factory()
            try:
                bench_one(name, model.fit_transform, make_transform(model), X, X_test, args.k)
            except Exception as exc:  # keep the sweep alive on reference quirks
                print(f"  {name:<24} FAILED: {exc}")


if __name__ == "__main__":
    main()
