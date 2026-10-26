"""Complete-episode GAE and clipped PPO over immutable behavior observations."""
from dataclasses import dataclass
import copy
import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class Transition:
    observation: object
    critic_state: np.ndarray
    action: int
    old_logp: float
    old_value: float
    reward: float
    terminal: bool


def gae(transitions, gamma, lam):
    if not transitions or not transitions[-1].terminal:
        raise ValueError("GAE expects complete episodes ending in a terminal transition")
    advantages = np.zeros(len(transitions), np.float32)
    carried = 0.
    for t in range(len(transitions) - 1, -1, -1):
        tr = transitions[t]
        next_value = 0 if tr.terminal else transitions[t + 1].old_value
        delta = tr.reward + gamma * next_value - tr.old_value
        carried = delta + gamma * lam * (not tr.terminal) * carried
        advantages[t] = carried
    returns = advantages + np.array([x.old_value for x in transitions], np.float32)
    return advantages, returns


def snapshot(observation):
    # In particular, preserve candidate ordering/masks and the recurrent history prefix.
    return copy.deepcopy(observation)


class PPO:
    def __init__(self, actor, critic, config):
        self.actor, self.critic, self.c = actor, critic, config
        self.parameters = list(actor.parameters()) + list(critic.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=config.learning_rate)

    def update(self, transitions, rng):
        if not transitions:
            return {}
        c = self.c
        device = next(self.actor.parameters()).device
        advantages, returns = gae(transitions, c.gamma, c.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        losses = []
        for _ in range(c.epochs):
            order = rng.permutation(len(transitions))
            for start in range(0, len(order), c.minibatch):
                idx = order[start:start + c.minibatch]
                batch = [transitions[i] for i in idx]
                logp, entropy = self.actor.evaluate([x.observation for x in batch], [x.action for x in batch])
                old = torch.tensor([x.old_logp for x in batch], device=device)
                adv = torch.as_tensor(advantages[idx], device=device)
                ratio = (logp - old).exp()
                policy = -torch.minimum(ratio * adv, ratio.clamp(1 - c.clip, 1 + c.clip) * adv).mean()
                states = torch.as_tensor(np.stack([x.critic_state for x in batch]), device=device)
                target = torch.as_tensor(returns[idx], device=device)
                value = F.mse_loss(self.critic(states), target)
                loss = policy + c.value_weight * value - c.entropy_weight * entropy.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO objective")
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters, c.grad_clip)
                self.optimizer.step()
                losses.append(float(loss.detach()))
        return {"loss": float(np.mean(losses)), "transitions": len(transitions)}

