import numpy as np
from csu.monitor import psi_samples, psi_shares


def test_psi_is_small_for_same_distribution_and_large_for_shift():
    r = np.random.default_rng(1)
    a, b, c = r.normal(size=5000), r.normal(size=5000), r.normal(1.5, 1, 5000)
    assert psi_samples(a, b) < .05
    assert psi_samples(a, c) > .25


def test_csi_shares():
    assert psi_shares({"a": .5, "b": .5}, {"a": .5, "b": .5}) < 1e-6
    assert psi_shares({"a": .5, "b": .5}, {"a": .9, "b": .1}) > .25
