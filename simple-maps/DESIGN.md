# Design notes

## Spec

1. Equal to or better runtime than existing implementations at inference
   time (willing to eat some training cost for large N and dim).
2. UMAP, TriMap, and PaCMAP.

Production goals: device interoperability (GPU/CPU/TPU), intuitive
save/load biased toward parametric formulations, and at most one main
backend dependency.

## v0 decisions

### Backend: pure NumPy, backend decision deferred

The Jax-vs-PyTorch question doesn't need answering to build the standard
(non-parametric) algorithms, so v0 doesn't answer it. Everything is written
as vectorized array programs — gathers, scatters (`np.add.at`), reductions —
which is exactly the subset that ports 1:1 to `jax.numpy` or `torch`. The
parametric phase picks the backend; the code shape is already right.

What we deliberately avoid: numba (JIT dependency), pynndescent/annoy
(dependency hell, the reason this project exists), scipy (only used
upstream for `curve_fit` and sparse matrices — both reimplemented here in
~40 lines).

### Inference: kNN interpolation, not per-query optimization

`transform` embeds a new point as the inverse-distance-weighted average of
the embeddings of its k nearest training points. That's one blocked kNN
query — microseconds per point — versus umap-learn's per-query SGD, which
is where its 12s/500-point transform goes. This is the main lever behind
spec goal #1 and it front-runs the parametric phase: a parametric model
replaces the interpolation with a forward pass, same interface.

### Save/load: one .npz file

`model.save(path)` writes config (JSON), training data (float32), and the
embedding (float32) into one compressed npz; `simple_maps.load(path)`
restores any estimator. No pickle: the format is inspectable, versioned
(`format_version`), and safe to load (`allow_pickle=False`). Parametric
models will store weights instead of training data under the same format.

### Algorithms

- **UMAP** (`umap_.py`): faithful to the paper/umap-learn — smooth-kNN
  bandwidth calibration (vectorized binary search), probabilistic t-conorm
  symmetrization keeping both edge directions, (a, b) curve fit done with
  Adam instead of scipy `curve_fit`, and per-epoch vectorized SGD with
  negative sampling instead of numba's per-edge loop. PCA init by default.
- **TriMap** (`trimap_.py`): reference triplet sampling (scaled
  similarities, inlier/outlier swap, weight_adj=500 log damping); deviates
  by optimizing with full-batch Adam rather than delta-bar-delta.
- **PaCMAP** (`pacmap_.py`): paper-exact pair sets (scaled-distance
  neighbors, second-closest-of-6 mid-near pairs, random further pairs) and
  the 100/100/250 three-phase weight schedule, Adam optimizer.

### kNN: exact and blocked

Brute force, computed in row blocks so memory stays at
`block_size * n` floats. Exact results make the algorithm implementations
easy to validate; an approximate index (NN-descent, no deps) slots in
behind the same `knn()` signature later. This is the main cost for large N
today — O(n^2 d) — and the first thing to replace.

## Known deviations / debts

- TriMap uses Adam (lr=0.1) instead of delta-bar-delta; quality matches on
  benchmarks but is slower than reference and is the slowest of the three.
- `np.add.at` is the scatter bottleneck in all three optimizers; a
  `bincount`-based scatter or the backend port is the known fix.
- UMAP negative sampling applies a fixed `negative_sample_rate` per sampled
  edge per epoch (umap-learn amortizes with a second per-edge schedule);
  effect is the same in expectation.
- `transform` interpolation cannot extrapolate beyond the training
  manifold. Acceptable for v0; solved properly by the parametric phase.

## Roadmap

1. ~~Core ops: kNN search, distance metrics~~ (v0)
2. ~~Standard UMAP, TriMap, PaCMAP + benchmark vs. existing~~ (v0)
3. Optimize training speed: bincount scatter, float32 path, approximate
   kNN for large N.
4. Parametric UMAP; choose the single backend (leaning Jax: `jax.numpy` is
   a near-drop-in for the current code and covers GPU/CPU/TPU) with NumPy
   kept as the no-dependency fallback.
