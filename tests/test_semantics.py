import numpy as np
import pytest
from setppo.semantics import encode, decode, quality, learn, project_simplex_rows


def test_support_and_normalized_positive_weights():
    s, beta, mask, error = encode([1, 0], np.eye(2), .2)
    np.testing.assert_allclose(s, [1, 0])
    np.testing.assert_allclose(beta, [1, 0])
    assert mask.tolist() == [True, False] and error == 0


def test_constrained_mastery_decoder_has_analytic_solution():
    p = np.eye(2)
    r, alpha = decode(np.array([1., 0.]), p, np.array([True, True]), np.array([.5, 1.]), .1)
    np.testing.assert_allclose(alpha, [1, 0], atol=1e-6)
    np.testing.assert_allclose(r, [.5, 0], atol=1e-6)
    assert quality(np.array([1., 0.]), r) == pytest.approx(.75)


def test_knowledge_improvement_cannot_worsen_optimal_reconstruction():
    p = np.eye(3)
    s = np.array([.6, .4, 0.])
    old, _ = decode(s, p, np.array([True, False, False]), np.array([.4, 0, 0]), .1)
    new, _ = decode(s, p, np.array([True, True, False]), np.array([.8, .9, 0]), .1)
    assert quality(s, new) >= quality(s, old) - 1e-8


def test_data_never_discovers_unavailable_atoms_but_feedback_can():
    chi, theta = np.array([False, True]), np.array([0., .2])
    learn(chi, theta, np.ones(2), .4)
    assert not chi[0] and theta[0] == 0
    assert theta[1] == pytest.approx(1 - .8 * np.exp(-.4))
    learn(chi, theta, np.array([1., 0]), .4, reveal=0)
    assert chi[0] and theta[0] > .1


def test_same_exposure_has_diminishing_mastery_gain():
    gains = []
    for initial in (.1, .8):
        theta = np.array([initial])
        learn(np.array([True]), theta, np.ones(1), .4)
        gains.append(theta[0] - initial)
    assert 0 < gains[1] < gains[0]


def test_simplex_projection():
    projected = project_simplex_rows(np.array([[3., 2., -1.], [.1, .2, -.3]]))
    assert np.all(projected >= 0) and np.all(projected.sum(1) <= 1 + 1e-8)
    np.testing.assert_allclose(projected[1], [.1, .2, 0])

