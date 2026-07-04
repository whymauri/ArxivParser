"""How the learned-classifier pipeline scales with training set size.

Pipeline under test, per implementation:
    fit on n points -> save to disk -> load -> predict 1000 held-out points

- simple-maps: KNNClassifier over UMAP (non-parametric) and ParametricUMAP,
  saved as a single .npz via model.save / simple_maps.load.
- umap-learn baseline: umap.UMAP + sklearn KNeighborsClassifier on the
  embedding, saved with pickle (the standard way to persist umap-learn).

Usage:
    python benchmarks/scaling_classifier.py [--sizes 1000 2500 5000 10000] [--dim 50]
"""

from __future__ import annotations

import argparse
import os
import pickle
import tempfile
import time

import numpy as np

from simple_maps import KNNClassifier, ParametricUMAP, UMAP, load
from simple_maps.datasets import gaussian_blobs

N_QUERY = 1000


def fmt_bytes(b: int) -> str:
    return f"{b / 1e6:8.2f}MB" if b >= 1e6 else f"{b / 1e3:8.1f}kB"


def run_simple_maps(embedder_factory, X, y, X_test, y_test, workdir):
    path = os.path.join(workdir, "clf.npz")
    clf = KNNClassifier(embedder_factory())

    t0 = time.perf_counter()
    clf.fit(X, y)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    clf.save(path)
    save_s = time.perf_counter() - t0
    size = os.path.getsize(path)

    t0 = time.perf_counter()
    restored = load(path)
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = restored.predict(X_test)
    predict_s = time.perf_counter() - t0
    acc = float(np.mean(pred == y_test))
    return fit_s, save_s, size, load_s, predict_s, acc


def run_umap_learn(X, y, X_test, y_test, workdir):
    import umap as umap_learn
    from sklearn.neighbors import KNeighborsClassifier

    path = os.path.join(workdir, "clf.pkl")

    t0 = time.perf_counter()
    reducer = umap_learn.UMAP(random_state=0)
    emb = reducer.fit_transform(X)
    knn_clf = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(emb, y)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    with open(path, "wb") as f:
        pickle.dump({"reducer": reducer, "clf": knn_clf}, f)
    save_s = time.perf_counter() - t0
    size = os.path.getsize(path)

    t0 = time.perf_counter()
    with open(path, "rb") as f:
        restored = pickle.load(f)
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = restored["clf"].predict(restored["reducer"].transform(X_test))
    predict_s = time.perf_counter() - t0
    acc = float(np.mean(pred == y_test))
    return fit_s, save_s, size, load_s, predict_s, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2500, 5000, 10000])
    ap.add_argument("--dim", type=int, default=50)
    args = ap.parse_args()

    header = f"  {'pipeline':<28} {'fit':>8} {'save':>8} {'size':>10} {'load':>8} {'predict':>9} {'acc':>6}"

    for n in args.sizes:
        X, y = gaussian_blobs(
            n_samples=n + N_QUERY, n_features=args.dim, n_clusters=10, random_state=0
        )
        perm = np.random.default_rng(0).permutation(len(X))
        X, y = X[perm], y[perm]
        X_train, y_train = X[:n], y[:n]
        X_test, y_test = X[n:], y[n:]

        print(f"\nn={n}, dim={args.dim}, {N_QUERY} queries")
        print(header)

        runners = [
            ("simple-maps UMAP+kNN", lambda w: run_simple_maps(
                lambda: UMAP(random_state=0), X_train, y_train, X_test, y_test, w)),
            ("simple-maps ParametricUMAP+kNN", lambda w: run_simple_maps(
                lambda: ParametricUMAP(random_state=0), X_train, y_train, X_test, y_test, w)),
        ]
        try:
            import umap  # noqa: F401

            runners.append(("umap-learn+sklearn (pickle)", lambda w: run_umap_learn(
                X_train, y_train, X_test, y_test, w)))
        except ImportError:
            print("  (umap-learn not installed; baseline skipped)")

        for name, runner in runners:
            with tempfile.TemporaryDirectory() as workdir:
                try:
                    fit_s, save_s, size, load_s, predict_s, acc = runner(workdir)
                    print(
                        f"  {name:<28} {fit_s:7.2f}s {save_s:7.3f}s {fmt_bytes(size)}"
                        f" {load_s:7.3f}s {predict_s:8.3f}s {acc:6.3f}"
                    )
                except Exception as exc:
                    print(f"  {name:<28} FAILED: {exc}")


if __name__ == "__main__":
    main()
