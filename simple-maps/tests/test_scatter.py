import numpy as np

from simple_maps._scatter import scatter_add


def test_scatter_add_matches_np_add_at():
    rng = np.random.default_rng(0)
    out = rng.normal(size=(50, 3))
    idx = rng.integers(0, 50, size=400)  # heavy duplication
    vals = rng.normal(size=(400, 3))

    expected = out.copy()
    np.add.at(expected, idx, vals)
    scatter_add(out, idx, vals)
    assert np.allclose(out, expected, atol=1e-12)


def test_scatter_add_untouched_rows_unchanged():
    out = np.ones((10, 2))
    scatter_add(out, np.array([3, 3, 7]), np.array([[1.0, 2.0], [1.0, 2.0], [5.0, 5.0]]))
    expected = np.ones((10, 2))
    expected[3] += [2.0, 4.0]
    expected[7] += [5.0, 5.0]
    assert np.allclose(out, expected)
