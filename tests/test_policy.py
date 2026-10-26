import copy
import numpy as np
import torch
from setppo.env import SemanticEnv
from setppo.networks import SetActor
from setppo.ppo import snapshot, gae, Transition


def test_joint_distribution_and_permutation_invariance(config, dataset):
    env = SemanticEnv(config, dataset)
    obs = env.reset(41)
    actor = SetActor(config)
    logp = actor.distributions([obs])[0]
    assert torch.allclose(logp.exp().sum(), torch.tensor(1.), atol=1e-6)
    permutation = np.arange(len(obs.items))[::-1].copy()
    shuffled = copy.deepcopy(obs)
    shuffled.items, shuffled.kinds = obs.items[permutation].copy(), obs.kinds[permutation].copy()
    shuffled.actions = tuple(obs.actions[i] for i in permutation)
    other = actor.distributions([shuffled])[0]
    torch.testing.assert_close(logp[torch.as_tensor(permutation)], other)


def test_behavior_snapshot_preserves_candidate_order_and_history(config, dataset):
    env = SemanticEnv(config, dataset)
    obs = env.reset(43)
    frozen = snapshot(obs)
    actor = SetActor(config)
    old = actor.distributions([frozen]).detach().clone()
    obs.items[:] = -123
    env.step(env.legal_actions()[0])
    torch.testing.assert_close(actor.distributions([frozen]), old)


def test_gae_resets_at_episode_boundaries():
    rows = [Transition(None, np.zeros(1), 0, 0, 0, 1., True),
            Transition(None, np.zeros(1), 0, 0, 0, 2., True)]
    adv, returns = gae(rows, .99, .95)
    np.testing.assert_allclose(adv, [1, 2])
    np.testing.assert_allclose(returns, [1, 2])

