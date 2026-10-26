"""Numerical forms of equations (2)--(8), with no policy decisions hidden here."""
import numpy as np
from scipy.optimize import minimize

EPS = 1e-10


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


def encode(z, prototypes, threshold):
    """Eqs. (2), (3): thresholded cosine support and positive normalized weights."""
    z = unit(z)
    cosines = prototypes @ z
    mask = (cosines >= threshold) & (cosines > 0)
    beta = np.where(mask, np.maximum(cosines, 0), 0.)
    if beta.sum() > 0:
        beta /= beta.sum()
    message = beta @ prototypes
    error = float(np.sum((z - message) ** 2) / (np.sum(z * z) + EPS))
    return message, beta, mask, error


def decode(message, prototypes, coverage, mastery, threshold):
    """Eq. (4): convex constrained least squares, alpha>=0 and sum(alpha)<=1.

    SLSQP convergence is checked; a failed solve is never silently called exact.
    """
    s = np.asarray(message, dtype=np.float64)
    candidates = np.flatnonzero(coverage & (prototypes @ unit(s) >= threshold))
    if not len(candidates):
        return np.zeros_like(s), np.zeros(len(prototypes))
    matrix = (prototypes[candidates] * mastery[candidates, None]).T
    def fun(a):
        residual = matrix @ a - s
        return residual @ residual
    def jac(a):
        return 2 * matrix.T @ (matrix @ a - s)
    opt = minimize(fun, np.full(len(candidates), 1 / len(candidates)), jac=jac,
                   method="SLSQP", bounds=[(0., 1.)] * len(candidates),
                   constraints={"type": "ineq", "fun": lambda a: 1 - a.sum(),
                                "jac": lambda a: -np.ones_like(a)},
                   options={"ftol": 1e-9, "maxiter": 200})
    if not opt.success:
        raise RuntimeError(f"Receiver decoder did not converge: {opt.message}")
    alpha = np.zeros(len(prototypes))
    alpha[candidates] = opt.x
    return matrix @ opt.x, alpha


def quality(message, reconstructed):
    """Eq. (5), evaluated BEFORE the subsequent knowledge update."""
    return float(max(0., 1 - np.sum((message - reconstructed) ** 2) /
                     (np.sum(message * message) + EPS)))


def learn(coverage, mastery, signal, rate, reveal=None, novel_mastery=.1):
    """Eq. (8): DATA cannot reveal an unavailable atom; KFB may reveal one."""
    if reveal is not None and not coverage[reveal]:
        coverage[reveal] = True
        mastery[reveal] = novel_mastery
    mastery[:] = np.where(coverage, 1 - (1 - mastery) * np.exp(-rate * signal), 0.)


def project_simplex_rows(x):
    """Euclidean projection onto nonnegative rows with sum<=1."""
    positive = np.maximum(x, 0.)
    needs = positive.sum(axis=-1) > 1
    if not np.any(needs):
        return positive
    v = x[needs]
    u = np.sort(v, axis=-1)[:, ::-1]
    css = np.cumsum(u, axis=-1) - 1
    k = np.arange(1, x.shape[-1] + 1)
    rho = np.maximum((u - css / k > 0).sum(axis=-1) - 1, 0)
    theta = css[np.arange(len(v)), rho] / (rho + 1)
    positive[needs] = np.maximum(v - theta[:, None], 0.)
    return positive


def particle_reconstruction(message, prototypes, coverage, mastery, threshold, iterations):
    """Batched approximate decoder ONLY for the uncertainty filter, not Eq. (4)."""
    enabled = coverage & (prototypes @ unit(message) >= threshold)[None, :]
    theta = mastery * enabled
    alpha = np.zeros_like(theta)
    # Frobenius norm squared is a safe Lipschitz upper bound for each particle.
    step = 1 / np.maximum(2 * np.sum(theta * theta, axis=1), EPS)
    for _ in range(iterations):
        predicted = (alpha * theta) @ prototypes
        grad = 2 * ((predicted - message) @ prototypes.T) * theta
        alpha = project_simplex_rows(alpha - step[:, None] * grad) * enabled
    return (alpha * theta) @ prototypes

