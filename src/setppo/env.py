"""DATA/KREQ/KFB simulator with a fixed task population and an explicit access boundary."""
from dataclasses import dataclass
import hashlib
import numpy as np
from .belief import Belief, sample_knowledge
from .semantics import decode, quality, learn, EPS


@dataclass(frozen=True)
class Action:
    kind: int  # 0=DATA, 1=KFB; there is no idle action.
    task: int
    atom: int = -1


@dataclass
class Observation:
    items: np.ndarray
    kinds: np.ndarray
    global_features: np.ndarray
    history: np.ndarray
    actions: tuple[Action, ...]


@dataclass
class EveObservation:
    features: np.ndarray
    history: np.ndarray
    can_observe: bool


class SemanticEnv:
    def __init__(self, config, dataset, projection_seed=0):
        self.c, self.data = config, dataset
        if dataset.prototypes.shape != (config.atoms, config.embedding_dim):
            raise ValueError("Prepared dictionary dimensions do not match the configuration")
        if dataset.metadata["kind"] != config.data_kind:
            raise ValueError("A synthetic smoke dataset must not be labeled as MIND")
        rng = np.random.default_rng(projection_seed)
        self.projection = rng.normal(0, 1 / np.sqrt(config.residual_dim),
                                     size=(config.residual_dim, config.embedding_dim))

    def reset(self, seed, split="train"):
        # Independent streams ensure Eve choices cannot perturb legitimate transitions.
        streams = np.random.SeedSequence(seed).spawn(3)
        rng = np.random.default_rng(streams[0])
        eligible = self.data.indices(split)
        pairs = len(eligible) * self.c.readers
        if pairs < self.c.tasks:
            raise ValueError("Not enough eligible news-reader pairs in this partition")
        chosen = rng.choice(pairs, self.c.tasks, replace=False)
        self.news = eligible[chosen // self.c.readers]
        self.readers = chosen % self.c.readers
        self.chi, self.theta = sample_knowledge(rng, (self.c.readers, self.c.atoms), .35, .70, (2, 2))
        self.eve_chi, self.eve_theta = sample_knowledge(
            np.random.default_rng(streams[1]), (self.c.atoms,), .15, .35, (2, 5))
        self.eve_initial_chi, self.eve_initial_theta = self.eve_chi.copy(), self.eve_theta.copy()
        self.belief = Belief(self.c, self.data.prototypes, self.projection, streams[2])
        self.completed = np.zeros(self.c.tasks, dtype=bool)
        self.blocked = np.zeros(self.c.tasks, dtype=bool)
        self.repairs, self.waiting = {}, {}
        self.failures = np.zeros(self.c.tasks, dtype=int)
        self.history, self.eve_history = [], []
        self.t, self.budget, self.kfb_count = 0, self.c.observer_budget, 0
        self.localization_hits, self.localization_count = 0, 0
        self.trace = []
        self.pair_fingerprint = hashlib.sha256(
            np.column_stack([self.news, self.readers]).tobytes() + self.chi.tobytes() + self.theta.tobytes()
        ).hexdigest()
        return self.legitimate_view()

    def legal_actions(self):
        if self.t >= self.c.horizon or self.completed.all():
            return ()
        pending = set(self.repairs) | set(self.waiting)
        data = [Action(0, i) for i in range(self.c.tasks)
                if not self.completed[i] and not self.blocked[i] and i not in pending]
        feedback = [Action(1, i, atom) for i, atom in sorted(self.repairs.items())]
        return tuple(data + feedback)

    @property
    def done(self):
        return not self.legal_actions()

    def _history(self, eve=False):
        history = self.eve_history if eve else self.history
        return np.stack(history).astype(np.float32) if history else np.zeros((1, self.c.event_dim), np.float32)

    def legitimate_view(self):
        actions = self.legal_actions()
        features = []
        for a in actions:
            reader, article = self.readers[a.task], self.news[a.task]
            coverage, mastery, missing, gap = self.belief.stats(reader)
            beta = self.data.beta[article]
            if a.kind == 0:
                descriptor = self.data.messages[article]
                stats = [beta @ coverage, beta @ mastery, beta @ gap,
                         self.failures[a.task] / self.c.horizon, 1 - self.t / self.c.horizon]
            else:
                descriptor = self.data.prototypes[a.atom]
                stats = [coverage[a.atom], mastery[a.atom], missing[a.atom], gap[a.atom],
                         self.failures[a.task] / self.c.horizon]
            features.append(np.r_[np.eye(2)[a.kind], np.eye(self.c.readers)[reader], descriptor, stats])
        items = np.array(features, np.float32).reshape(-1, self.c.item_dim)
        public = np.array([self.t / self.c.horizon, self.completed.mean(),
                           len(self.repairs) / self.c.feedback_capacity, self.blocked.mean()], np.float32)
        return Observation(items.copy(), np.array([a.kind for a in actions]), public,
                           self._history(), actions)

    def observer_view(self, committed_action):
        metadata = np.zeros(2 + self.c.readers)
        if self.c.observer_class in {"E1", "E2"}:
            metadata[:2] = np.eye(2)[committed_action.kind]
            metadata[2:] = np.eye(self.c.readers)[self.readers[committed_action.task]]
        features = np.r_[self.eve_chi, self.eve_theta,
                         self.budget / max(self.c.observer_budget, 1),
                         self.t / self.c.horizon, metadata].astype(np.float32)
        # No candidate item ID, message content, true reader state or semantic status.
        return EveObservation(features, self._history(eve=True), self.budget > 0)

    def critic_state(self):
        """Training-only privileged input; never fed to either actor."""
        return np.r_[self.chi.ravel(), self.theta.ravel(), self.eve_chi, self.eve_theta,
                     self.completed, self.t / self.c.horizon, self.budget / max(self.c.observer_budget, 1),
                     len(self.repairs) / self.c.feedback_capacity, len(self.waiting) / self.c.tasks,
                     self.blocked.mean()].astype(np.float32)

    def _event(self, action, success, residual, observed, eve=False):
        reader, article = self.readers[action.task], self.news[action.task]
        metadata_visible = not eve or self.c.observer_class in {"E1", "E2"}
        kind = np.eye(2)[action.kind] if metadata_visible or observed else np.zeros(2)
        receiver = np.eye(self.c.readers)[reader] if metadata_visible else np.zeros(self.c.readers)
        content = self.data.messages[article] if action.kind == 0 else self.data.prototypes[action.atom]
        if eve and not observed:
            content = np.zeros(self.c.embedding_dim)
        q = residual if not eve or self.c.observer_class == "E2" else np.zeros(self.c.residual_dim)
        status = float(observed) if eve else float(success)
        return np.r_[kind, receiver, content, status, q]

    def _activate_waiting(self):
        # Stable task order is an explicit implementation choice; no task is dropped.
        while self.waiting and len(self.repairs) < self.c.feedback_capacity:
            task = min(self.waiting)
            self.repairs[task] = self.waiting.pop(task)

    def step(self, action, observe=False):
        if action not in self.legal_actions():
            raise ValueError("Action is not executable")
        if observe and self.budget <= 0:
            raise ValueError("Observer budget exhausted")
        c = self.c
        reader, article = self.readers[action.task], self.news[action.task]
        beta = self.data.beta[article]
        reader_before = self.theta[reader].copy()
        before = self.eve_theta.copy()
        success = False
        residual = np.zeros(c.residual_dim)
        if action.kind == 0:
            message = self.data.messages[article]
            reconstructed, _ = decode(message, self.data.prototypes, self.chi[reader],
                                      self.theta[reader], c.tau_receiver)
            # The completion decision uses the pre-learning state, Eq. (5).
            success = quality(message, reconstructed) >= c.tau_decoder
            if success:
                self.completed[action.task] = True
            else:
                self.failures[action.task] += 1
                residual = self.projection @ (message - reconstructed)
            self.belief.data_update(reader, message, beta, success, residual if not success else None)
            learn(self.chi[reader], self.theta[reader], beta, c.eta_reader)
            if not success:
                if c.allow_feedback and np.any(beta > 0):
                    if c.selector == "oracle":
                        # This privileged selector is ONLY the explicitly named Oracle diagnostic.
                        atom = int(np.argmax(beta * (1 - self.theta[reader])))
                    else:
                        atom = self.belief.select(reader, beta, residual)
                    self.repairs[action.task] = atom if len(self.repairs) < c.feedback_capacity else -1
                    if self.repairs[action.task] < 0:
                        del self.repairs[action.task]
                        self.waiting[action.task] = atom
                    oracle = int(np.argmax(beta * (1 - self.theta[reader])))
                    self.localization_hits += atom == oracle
                    self.localization_count += 1
                else:
                    self.blocked[action.task] = True
            signal, reveal = beta, None
        else:
            signal = np.eye(c.atoms)[action.atom]
            reveal = action.atom
            learn(self.chi[reader], self.theta[reader], signal, c.eta_reader,
                  reveal, c.initial_novel_mastery)
            self.belief.feedback_update(reader, action.atom)
            del self.repairs[action.task]
            self.kfb_count += 1
            # A KFB action never marks a DATA task completed.
            self._activate_waiting()
        changed_atoms = self.theta[reader] > reader_before + EPS
        affected = (self.readers == reader) & (self.data.beta[self.news] @ changed_atoms > 0)
        self.blocked[affected] = False
        # The just-failed task without repair remains blocked until a subsequent update.
        if action.kind == 0 and not success and not c.allow_feedback:
            self.blocked[action.task] = True
        if observe:
            self.budget -= 1
            learn(self.eve_chi, self.eve_theta, signal, c.eta_eve, reveal,
                  c.initial_novel_mastery)
        growth = float(np.sum(self.eve_theta - before))
        self.history.append(self._event(action, success, residual, observe))
        self.eve_history.append(self._event(action, success, residual, observe, eve=True))
        self.t += 1
        self.trace.append({"step": self.t, "type": action.kind, "task": action.task,
                           "atom": action.atom, "completion": int(success),
                           "observed": bool(observe), "growth": growth})
        assert self.completed.sum() + (~self.completed).sum() == c.tasks
        assert len(self.repairs) <= c.feedback_capacity and 0 <= self.budget <= c.observer_budget
        return {"growth": growth, "mu": int(success), "done": self.done}

    def metrics(self):
        complete = int(self.completed.sum())
        tcr = complete / self.c.tasks
        return {"tcr": tcr, "episode_feasible": tcr >= self.c.service_target,
                "completed": complete, "required": self.c.tasks,
                "growth": float(np.sum(self.eve_theta - self.eve_initial_theta)),
                "normalized_growth": float(np.mean(self.eve_theta - self.eve_initial_theta)),
                "discovery": int(np.sum(self.eve_chi & ~self.eve_initial_chi)),
                "kfb_messages": self.kfb_count,
                "kfb_per_completed": self.kfb_count / complete if complete else None,
                "localization_hits": int(self.localization_hits),
                "localization_count": self.localization_count,
                "observations": self.c.observer_budget - self.budget,
                "steps": self.t, "pair_fingerprint": self.pair_fingerprint}

