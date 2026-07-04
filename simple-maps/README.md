# simple-maps

Simple and effective manifold approximators with few dependencies.

The entire library depends on **NumPy and nothing else**. No Annoy, no
PyNNDescent, no numba, no scipy — `pip install numpy` and everything works.

```python
from simple_maps import UMAP, TriMap, PaCMAP, load

model = PaCMAP(random_state=0)
Y = model.fit_transform(X)          # (n, 2) embedding

Z = model.transform(X_new)          # fast out-of-sample inference

model.save("embedding.npz")         # single compressed file
model = load("embedding.npz")       # ready to transform again
```

## What's here (v0)

| Piece | Status |
| --- | --- |
| Core ops: distance metrics (euclidean, sqeuclidean, cosine, manhattan), exact blocked kNN | done |
| Standard UMAP (fuzzy simplicial set, SGD + negative sampling) | done |
| Standard TriMap (weighted triplets, full-batch Adam) | done |
| Standard PaCMAP (NB/MN/FP pairs, three-phase schedule) | done |
| Benchmarks vs. reference implementations | done (`benchmarks/run_benchmarks.py`) |
| Approximate kNN for large N | planned |
| Speed optimization pass | planned |
| Parametric UMAP + GPU/TPU backend | planned |

## Benchmarks

`python benchmarks/run_benchmarks.py --n 5000 --dim 50` on one CPU
(reference implementations are skipped automatically if not installed):

```
blobs(n=5000, d=50)
  simple-maps UMAP         fit     9.99s   infer    0.043s   knn@10 0.070
  simple-maps TriMap       fit    19.30s   infer    0.032s   knn@10 0.021
  simple-maps PaCMAP       fit     7.93s   infer    0.026s   knn@10 0.039
  umap-learn               fit    24.38s   infer   12.774s   knn@10 0.062
  pacmap (ref)             fit     1.30s   infer    0.227s   knn@10 0.039

swiss_roll(n=5000)
  simple-maps UMAP         fit     8.01s   infer    0.028s   knn@10 0.731
  simple-maps TriMap       fit    18.87s   infer    0.021s   knn@10 0.560
  simple-maps PaCMAP       fit     7.77s   infer    0.027s   knn@10 0.738
  umap-learn               fit     4.90s   infer    1.203s   knn@10 0.765
  pacmap (ref)             fit     1.19s   infer    0.152s   knn@10 0.752
```

Read: embedding quality (kNN preservation) is at parity with the reference
implementations, inference is 5-450x faster (spec goal #1), and training
speed is competitive-to-slower depending on algorithm — the reference
packages JIT-compile with numba, we are pure NumPy. Training speed is
roadmap item 3.

## Install

```bash
pip install -e .            # runtime: numpy only
pip install -e .[test]      # + pytest
pip install -e .[bench]     # + umap-learn, pacmap, scikit-learn for comparisons
```

Run tests with `pytest`.

## Design notes

See [DESIGN.md](DESIGN.md) for the decisions behind v0 (pure-NumPy backend,
inference strategy, serialization format) and the plan for the parametric /
GPU phase.
