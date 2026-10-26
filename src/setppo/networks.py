"""Masked hierarchical Deep Sets actor, recurrent history, and separate privileged critics."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def histories(observations, device):
    lengths = [len(o.history) for o in observations]
    x = np.zeros((len(observations), max(lengths), observations[0].history.shape[-1]), np.float32)
    for i, o in enumerate(observations):
        x[i, :lengths[i]] = o.history
    return torch.as_tensor(x, device=device), lengths


class History(nn.Module):
    def __init__(self, event_dim, hidden):
        super().__init__()
        self.gru = nn.GRU(event_dim, hidden, batch_first=True)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        return self.gru(packed)[1][-1]


class SetActor(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        h = c.hidden_dim
        self.phi = nn.Sequential(nn.Linear(c.item_dim, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.rho = nn.Sequential(nn.Linear(h, h), nn.Tanh())
        self.ordered_pool = nn.Linear((c.tasks + c.feedback_capacity) * h, h) if not c.set_encoder else None
        self.history = History(c.event_dim, h)
        self.context = nn.Sequential(nn.Linear(2 * h + 4, h), nn.Tanh())
        self.type_head = nn.Linear(h, 2)
        self.item_head = nn.Sequential(nn.Linear(2 * h, h), nn.Tanh(), nn.Linear(h, 1))

    def distributions(self, observations):
        device = next(self.parameters()).device
        n = max(len(o.items) for o in observations)
        if n == 0 or any(len(o.items) == 0 for o in observations):
            raise ValueError("No actor distribution exists for an empty executable set")
        features = np.zeros((len(observations), n, self.c.item_dim), np.float32)
        kinds = np.full((len(observations), n), -1, np.int64)
        for i, o in enumerate(observations):
            features[i, :len(o.items)] = o.items
            kinds[i, :len(o.kinds)] = o.kinds
        x, kinds = torch.as_tensor(features, device=device), torch.as_tensor(kinds, device=device)
        mask = kinds >= 0
        embedded = self.phi(x)
        if self.c.set_encoder:
            pool = self.rho((embedded * mask[..., None]).sum(1) / mask.sum(1, keepdim=True))
        else:
            slots = self.c.tasks + self.c.feedback_capacity
            ordered = F.pad(embedded * mask[..., None], (0, 0, 0, slots - n))
            pool = torch.tanh(self.ordered_pool(ordered.flatten(1)))
        hx, lengths = histories(observations, device)
        recurrent = self.history(hx, lengths)
        public = torch.as_tensor(np.stack([o.global_features for o in observations]), device=device)
        context = self.context(torch.cat([pool, recurrent, public], dim=-1))
        item_logits = self.item_head(torch.cat([embedded, context[:, None].expand(-1, n, -1)], -1)).squeeze(-1)
        if not self.c.hierarchical:
            return F.log_softmax(item_logits.masked_fill(~mask, -1e9), -1)
        type_mask = torch.stack([(kinds == t).any(1) for t in (0, 1)], -1)
        type_logp = F.log_softmax(self.type_head(context).masked_fill(~type_mask, -1e9), -1)
        joint = torch.full_like(item_logits, -1e9)
        for t in (0, 1):
            selected = kinds == t
            conditional = F.log_softmax(item_logits.masked_fill(~selected, -1e9), -1)
            joint = torch.where(selected, type_logp[:, t, None] + conditional, joint)
        return joint

    def evaluate(self, observations, actions):
        logp = self.distributions(observations)
        indices = torch.as_tensor(actions, device=logp.device, dtype=torch.long)
        return logp.gather(1, indices[:, None]).squeeze(1), -(logp.exp() * logp).sum(1)

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        logp = self.distributions([observation])[0]
        index = int(logp.argmax()) if deterministic else int(torch.distributions.Categorical(logits=logp).sample())
        return index, float(logp[index])


class ObserverActor(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = c.hidden_dim
        self.history = History(c.event_dim, h)
        self.head = nn.Sequential(nn.Linear(c.eve_dim + h, h), nn.Tanh(), nn.Linear(h, 2))

    def distributions(self, observations):
        device = next(self.parameters()).device
        x, lengths = histories(observations, device)
        own = torch.as_tensor(np.stack([o.features for o in observations]), device=device)
        logits = self.head(torch.cat([own, self.history(x, lengths)], -1))
        can = torch.as_tensor([o.can_observe for o in observations], device=device)
        logits = logits.clone()
        logits[:, 1] = logits[:, 1].masked_fill(~can, -1e9)
        return F.log_softmax(logits, -1)

    def evaluate(self, observations, actions):
        logp = self.distributions(observations)
        a = torch.as_tensor(actions, device=logp.device, dtype=torch.long)
        return logp.gather(1, a[:, None]).squeeze(1), -(logp.exp() * logp).sum(1)

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        logp = self.distributions([observation])[0]
        a = int(logp.argmax()) if deterministic else int(torch.distributions.Categorical(logits=logp).sample())
        return a, float(logp[a])


class Critic(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(c.critic_dim, c.hidden_dim), nn.Tanh(),
                                   nn.Linear(c.hidden_dim, c.hidden_dim), nn.Tanh(), nn.Linear(c.hidden_dim, 1))

    def forward(self, privileged_state):
        return self.value(privileged_state).squeeze(-1)

