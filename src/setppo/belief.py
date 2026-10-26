"""History-only particle belief; true receiver states are not accepted as inputs."""
import numpy as np
from .semantics import learn, particle_reconstruction, EPS


def sample_knowledge(rng, shape, low, high, beta):
    probabilities = rng.uniform(low, high, size=shape[:-1] + (1,))
    coverage = rng.random(shape) < probabilities
    mastery = rng.beta(*beta, size=shape) * coverage
    return coverage, mastery


class Belief:
    def __init__(self, config, prototypes, projection, seed):
        self.c, self.prototypes, self.projection = config, prototypes, projection
        self.rng = np.random.default_rng(seed)
        shape = (config.readers, config.belief_particles, config.atoms)
        self.coverage, self.mastery = sample_knowledge(self.rng, shape, .35, .70, (2, 2))
        self.weights = np.full(shape[:2], 1 / config.belief_particles)

    def stats(self, reader):
        weights = self.weights[reader]
        theta = self.mastery[reader]
        return (weights @ self.coverage[reader], weights @ theta,
                weights @ (theta < self.c.tau_known), weights @ (1 - theta))

    def data_update(self, reader, message, beta, success, residual):
        if not self.c.use_belief:
            return
        predicted = particle_reconstruction(
            message, self.prototypes, self.coverage[reader], self.mastery[reader],
            self.c.tau_receiver, self.c.belief_decoder_iterations)
        error = np.sum((message - predicted) ** 2, axis=1)
        q_pred = np.maximum(0, 1 - error / (message @ message + EPS))
        agrees = (q_pred >= self.c.tau_decoder) == success
        log_w = np.log(self.weights[reader] + EPS) + np.log(np.where(agrees, .95, .05))
        if residual is not None and self.c.selector != "posterior":
            difference = (message - predicted) @ self.projection.T - residual
            log_w -= np.sum(difference * difference, axis=1) / (2 * self.c.belief_noise ** 2)
        weights = np.exp(log_w - log_w.max())
        weights /= weights.sum()
        if 1 / np.sum(weights * weights) < self.c.belief_particles / 2:
            indices = self.rng.choice(len(weights), size=len(weights), p=weights)
            self.coverage[reader] = self.coverage[reader, indices]
            self.mastery[reader] = self.mastery[reader, indices]
            weights = np.full_like(weights, 1 / len(weights))
        self.weights[reader] = weights
        # Publicly known message and transition model, never hidden receiver labels.
        for chi, theta in zip(self.coverage[reader], self.mastery[reader]):
            learn(chi, theta, beta, self.c.eta_reader)

    def feedback_update(self, reader, atom):
        if not self.c.use_belief:
            return
        signal = np.eye(self.c.atoms)[atom]
        for chi, theta in zip(self.coverage[reader], self.mastery[reader]):
            learn(chi, theta, signal, self.c.eta_reader, atom, self.c.initial_novel_mastery)

    def select(self, reader, beta, residual):
        """Eq. (7) plus the explicitly described expected weighted-gap fallback."""
        _, _, missing, gap = self.stats(reader)
        support = np.flatnonzero(beta > 0)
        if not len(support):
            raise ValueError("A repair must use an eligible DATA support")
        if self.c.selector == "random":
            return int(self.rng.choice(support))
        if self.c.selector == "posterior" or np.linalg.norm(residual) < self.c.residual_confidence:
            return int(support[np.argmax(beta[support] * gap[support])])
        q = self.rng.permutation(residual) if self.c.selector == "shuffled" else residual
        projected = self.prototypes[support] @ self.projection.T
        cosine = projected @ q / np.maximum(np.linalg.norm(projected, axis=1) * np.linalg.norm(q), EPS)
        score = missing[support] * np.maximum(cosine, 0)
        if score.max() <= 0:
            score = beta[support] * gap[support]
        return int(support[np.argmax(score)])

