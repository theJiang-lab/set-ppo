"""Train on train IDs, select on validation IDs, evaluate paired held-out cases."""
from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import torch
from .config import Config, UNREPORTED_FIELDS
from .env import SemanticEnv
from .networks import SetActor, ObserverActor, Critic
from .ppo import PPO, Transition, snapshot
from .metrics import summarize
from .baselines import heuristic


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def models(c, device="cpu"):
    return [cls(c).to(device) for cls in (SetActor, ObserverActor, Critic, Critic)]


def load_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    c = Config(**saved["config"])
    nets = models(c, device)
    for net, name in zip(nets, ("legitimate", "observer", "legitimate_critic", "observer_critic")):
        net.load_state_dict(saved[name])
        net.eval()
    return c, nets, saved


def save_checkpoint(path, c, nets, iteration, dual, selection):
    saved = {"config": asdict(c), "iteration": iteration, "dual": float(dual),
             "selection": selection, "implementation": "manuscript_derived_reimplementation"}
    for net, name in zip(nets, ("legitimate", "observer", "legitimate_critic", "observer_critic")):
        saved[name] = net.state_dict()
    torch.save(saved, path)


def rollout(env, nets, dual, seed, split, observer_mode="reference", policy=None,
            deterministic=False):
    seed_all(seed)
    rng = np.random.default_rng(seed + 859)
    legitimate, eve, critic_l, critic_e = nets
    device = next(legitimate.parameters()).device
    env.reset(seed, split)
    legitimate_rows, observer_rows = [], []
    while not env.done:
        obs = env.legitimate_view()
        state = env.critic_state()
        with torch.no_grad():
            if policy is None:
                index, logp_l = legitimate.act(obs, deterministic)
            elif isinstance(policy, str):
                index, logp_l = heuristic(obs, policy, rng), 0.
            else:
                index, logp_l = int(policy.select_action(snapshot(obs))), 0.
            if not 0 <= index < len(obs.actions):
                raise ValueError("Policy selected an invalid candidate index")
            action = obs.actions[index]
            eve_obs = env.observer_view(action)
            if observer_mode == "reference":
                probability = min(1., env.budget / max(1, env.c.horizon - env.t))
                observed = int(rng.random() < probability) if eve_obs.can_observe else 0
                logp_e = float(np.log(max(probability if observed else 1 - probability, 1e-12)))
            elif observer_mode == "neural":
                observed, logp_e = eve.act(eve_obs, deterministic)
            else:
                raise ValueError("Use reference or neural observer_mode")
            tensor = torch.as_tensor(state, device=device).unsqueeze(0)
            value_l, value_e = float(critic_l(tensor)[0]), float(critic_e(tensor)[0])
        result = env.step(action, bool(observed))
        # Undiscounted physical semantic increments; discount is applied only by GAE.
        reward_l = -result["growth"] + dual * result["mu"] / env.c.tasks
        legitimate_rows.append(Transition(snapshot(obs), state.copy(), index, logp_l,
                                           value_l, reward_l, result["done"]))
        observer_rows.append(Transition(snapshot(eve_obs), state.copy(), observed,
                                         logp_e, value_e, result["growth"], result["done"]))
    metrics = env.metrics()
    metrics.update({"case_seed": seed, "split": split})
    return metrics, legitimate_rows, observer_rows


def evaluate_cases(c, dataset, nets, seed, episodes, split="test", observer_mode="reference",
                   policy=None, deterministic=False):
    env = SemanticEnv(c, dataset, projection_seed=seed)
    offset = {"train": 0, "val": 1_000_000_000, "test": 2_000_000_000}[split]
    return [rollout(env, nets, 0., offset + seed * 100_000 + i, split,
                    observer_mode, policy, deterministic)[0] for i in range(episodes)]


def train(c, dataset, output, seed=0, device="cpu", frozen_legitimate=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    rng = np.random.default_rng(seed)
    if frozen_legitimate:
        original, nets, _ = load_checkpoint(frozen_legitimate, device)
        if (original.atoms, original.embedding_dim, original.tasks, original.readers) != (
                c.atoms, c.embedding_dim, c.tasks, c.readers):
            raise ValueError("Adaptation must preserve the legitimate model dimensions")
        for parameter in nets[0].parameters():
            parameter.requires_grad_(False)
    else:
        nets = models(c, device)
    env = SemanticEnv(c, dataset, projection_seed=seed)
    ppo_l, ppo_e = PPO(nets[0], nets[2], c), PPO(nets[1], nets[3], c)
    dual, best = 0., None
    c.save(output / "resolved_config.json")
    manifest = {"seed": seed, "python": platform.python_version(), "numpy": np.__version__,
                "torch": torch.__version__, "dataset": dataset.metadata,
                "unreported_choices": {k: getattr(c, k) for k in UNREPORTED_FIELDS},
                "seed_ids_are_implementation_choices": True,
                "data_kind": c.data_kind, "published_results_reproduced": False,
                "mode": "adapt_observer_only" if frozen_legitimate else "joint_training"}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output / "training.jsonl").open("w", encoding="utf-8") as log:
        for iteration in range(c.iterations):
            rows_l, rows_e, metrics = [], [], []
            mode = "neural" if c.adversarial or frozen_legitimate else "reference"
            for episode in range(c.episodes_per_iteration):
                case = seed * 10_000_000 + iteration * c.episodes_per_iteration + episode
                m, l, e = rollout(env, nets, dual, case, "train", mode)
                rows_l.extend(l)
                rows_e.extend(e)
                metrics.append(m)
            summary = summarize(metrics)
            stats = {"iteration": iteration + 1, "train": summary}
            if not frozen_legitimate:
                stats["ppo_legitimate"] = ppo_l.update(rows_l, rng)
                # Complete-episode mean TCR, with projection, Eq. (A.3).
                dual = float(np.clip(dual + c.dual_lr *
                                     (c.service_target - summary["mean_tcr_pct"] / 100), 0, c.dual_max))
            if c.adversarial or frozen_legitimate:
                stats["ppo_observer"] = ppo_e.update(rows_e, rng)
            stats["dual"] = dual
            if (iteration + 1) % c.validation_interval == 0 or iteration + 1 == c.iterations:
                val = summarize(evaluate_cases(c, dataset, nets, seed, c.validation_episodes,
                                               "val", "neural" if frozen_legitimate else "reference"))
                stats["validation"] = val
                candidate = -val["mean_growth"] if frozen_legitimate else val["mean_growth"]
                feasible = frozen_legitimate is not None or val["mean_tcr_pct"] / 100 >= c.service_target
                if feasible and (best is None or candidate < best):
                    best = candidate
                    save_checkpoint(output / ("best_eve.pt" if frozen_legitimate else "best.pt"),
                                    c, nets, iteration + 1, dual,
                                    {"split": "val", "metrics": val, "mean_service_feasible": bool(feasible)})
            log.write(json.dumps(stats) + "\n")
            log.flush()
            print(f"iteration={iteration + 1} TCR={summary['mean_tcr_pct']:.2f} "
                  f"G={summary['mean_growth']:.4f} lambda={dual:.4f}", flush=True)
    save_checkpoint(output / "last.pt", c, nets, c.iterations, dual,
                    {"split": "train", "selected_on_validation": False})
    return output


def write_evaluation(path, rows, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"summary": summarize(rows), "metadata": metadata,
              "result_origin": "new_run_not_published_aggregate"}
    (path / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

