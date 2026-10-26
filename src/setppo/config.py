"""Reported parameters and explicitly identified, unreported implementation choices."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class Config:
    readers: int = 4
    tasks: int = 80
    horizon: int = 160
    atoms: int = 96
    embedding_dim: int = 16
    residual_dim: int = 10
    feedback_capacity: int = 3
    service_target: float = .85
    observer_budget: int = 20
    representation_error: float = .35
    learning_rate: float = 3e-4
    iterations: int = 2000
    episodes_per_iteration: int = 64
    minibatch: int = 256
    epochs: int = 4
    gamma: float = .99
    gae_lambda: float = .95
    clip: float = .2
    test_episodes: int = 500
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    # Values below are runnable choices, NOT values recovered from original runs.
    tau_source: float = .25
    tau_receiver: float = .2
    tau_decoder: float = .8
    tau_known: float = .6
    eta_reader: float = .8
    eta_eve: float = .3
    initial_novel_mastery: float = .1
    dual_lr: float = .05
    dual_max: float = 100.
    hidden_dim: int = 64
    belief_particles: int = 16
    belief_noise: float = .15
    belief_decoder_iterations: int = 40
    residual_confidence: float = .05
    entropy_weight: float = .01
    value_weight: float = .5
    grad_clip: float = .5
    validation_episodes: int = 20
    validation_interval: int = 10
    observer_class: str = "E0"
    reference_observer: str = "uniform_remaining_budget"
    selector: str = "residual"
    use_belief: bool = True
    hierarchical: bool = True
    set_encoder: bool = True
    adversarial: bool = True
    allow_feedback: bool = True
    data_kind: str = "mind"

    def __post_init__(self):
        if min(self.readers, self.tasks, self.horizon, self.atoms,
               self.embedding_dim, self.residual_dim, self.feedback_capacity) < 1:
            raise ValueError("Dimensions and episode sizes must be positive")
        if self.observer_class not in {"E0", "E1", "E2"}:
            raise ValueError("observer_class must be E0, E1 or E2")
        if not 0 <= self.observer_budget <= self.horizon:
            raise ValueError("Observer budget must be between zero and the horizon")
        if not 0 < self.service_target <= 1 or not 0 < self.tau_decoder <= 1:
            raise ValueError("Service and decoder thresholds must be in (0, 1]")
        if self.selector not in {"residual", "posterior", "shuffled", "random", "oracle"}:
            raise ValueError("Unknown selector")
        if self.data_kind not in {"mind", "synthetic_smoke"}:
            raise ValueError("Unknown dataset kind")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @property
    def item_dim(self):
        return 2 + self.readers + self.embedding_dim + 5

    @property
    def event_dim(self):
        return 2 + self.readers + self.embedding_dim + 1 + self.residual_dim

    @property
    def critic_dim(self):
        return 2 * (self.readers + 1) * self.atoms + self.tasks + 5

    @property
    def eve_dim(self):
        return 2 * self.atoms + 2 + 2 + self.readers


UNREPORTED_FIELDS = [
    "tau_source", "tau_receiver", "tau_decoder", "tau_known", "eta_reader",
    "eta_eve", "initial_novel_mastery", "dual_lr", "dual_max", "hidden_dim",
    "belief_particles", "belief_noise", "belief_decoder_iterations",
    "residual_confidence", "entropy_weight", "value_weight", "grad_clip",
    "validation_episodes", "validation_interval", "reference_observer",
]

