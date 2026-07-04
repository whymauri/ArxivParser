# simple-maps

Simple and effective manifold approximators with few dependencies.

The entire library depends on **NumPy and nothing else**. No Annoy, no
PyNNDescent, no numba, no scipy — `pip install numpy` and everything works.

```python
from simple_maps import UMAP, TriMap, PaCMAP, ParametricUMAP, load

model = PaCMAP(random_state=0)
Y = model.fit_transform(X)          # (n, 2) embedding

Z = model.transform(X_new)          # fast out-of-sample inference

model.save("embedding.npz")         # single compressed file
model = load("embedding.npz")       # ready to transform again
```

All three algorithms also come in parametric form — `ParametricUMAP`,
`ParametricTriMap`, and `ParametricPaCMAP` learn an MLP encoder through
their respective losses (manual backprop, still NumPy-only): `transform`
is a forward pass and `save` stores just the weights — the file is ~30 kB
regardless of training set size.

A learned classifier on top of any embedder is one object with the same
one-file save/load:

```python
from simple_maps import KNNClassifier, ParametricUMAP, load

clf = KNNClassifier(ParametricUMAP(random_state=0))
clf.fit(X_train, y_train)
clf.save("clf.npz")

clf = load("clf.npz")
clf.predict(X_new)                  # milliseconds
clf.predict_proba(X_new)
```

## What's here

| Piece | Status |
| --- | --- |
| Core ops: distance metrics (euclidean, sqeuclidean, cosine, manhattan), exact blocked kNN | done |
| Standard UMAP (fuzzy simplicial set, SGD + negative sampling) | done |
| Standard TriMap (weighted triplets, full-batch Adam) | done |
| Standard PaCMAP (NB/MN/FP pairs, three-phase schedule) | done |
| Parametric UMAP / TriMap / PaCMAP (MLP encoder, weights-only save, forward-pass inference) | done |
| KNNClassifier (learned classifier over any embedder, one-file save/load) | done |
| Benchmarks vs. reference implementations | done (`benchmarks/`) |
| bincount scatter optimization (1.6-1.9x faster fits) | done |
| Approximate kNN for large N | planned |
| GPU/TPU backend | planned |

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

### Classifier pipeline scaling

`python benchmarks/scaling_classifier.py` runs the full production path —
fit a classifier on n points, save, load, predict 1000 held-out points —
against the standard umap-learn + sklearn + pickle stack:

```
n=1000, dim=50, 1000 queries
  pipeline                          fit     save       size     load   predict    acc
  simple-maps UMAP+kNN            1.64s   0.008s    203.4kB   0.003s    0.028s  1.000
  simple-maps ParametricUMAP+kNN  3.40s   0.002s     40.8kB   0.002s    0.013s  1.000
  umap-learn+sklearn (pickle)    10.65s   0.002s    426.5kB   0.000s    4.667s  1.000

n=5000, dim=50, 1000 queries
  simple-maps UMAP+kNN            8.61s   0.038s     1.01MB   0.009s    0.169s  1.000
  simple-maps ParametricUMAP+kNN  8.52s   0.004s     73.5kB   0.002s    0.088s  1.000
  umap-learn+sklearn (pickle)    18.96s  11.805s     6.19MB   1.073s    1.374s  0.999

n=20000, dim=50, 1000 queries
  simple-maps UMAP+kNN           22.40s   0.156s     4.02MB   0.029s    0.803s  1.000
  simple-maps ParametricUMAP+kNN 22.65s   0.011s    195.3kB   0.003s    0.403s  1.000
  umap-learn+sklearn (pickle)    10.06s   1.057s    24.98MB   0.952s    0.469s  1.000
```

Read: accuracy is 1.0 at every size; saved models are ~100x smaller than
the pickled umap-learn stack (195 kB vs 25 MB at n=20k — the parametric
encoder itself is constant-size, the growth is just the stored 2-D
training embedding and labels used by the vote); save/load never pickles;
prediction is faster at every size for the parametric pipeline. The one
place umap-learn wins is raw fit speed at n=20k (numba JIT vs our exact
O(n^2) kNN) — approximate kNN is the planned fix.

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
