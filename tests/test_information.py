from dataclasses import replace
import numpy as np
from setppo.env import SemanticEnv, Action


def test_e0_has_no_current_message_or_recipient_metadata(config, dataset):
    env = SemanticEnv(config, dataset)
    env.reset(17)
    actions = env.legal_actions()
    a, b = actions[0], actions[-1]
    np.testing.assert_array_equal(env.observer_view(a).features, env.observer_view(b).features)


def test_e1_exposes_type_and_reader_but_not_current_content(config, dataset):
    env = SemanticEnv(replace(config, observer_class='E1'), dataset)
    env.reset(17)
    a = Action(0, 0)
    same_reader = np.flatnonzero(env.readers == env.readers[0])
    b = Action(0, int(same_reader[-1]))
    np.testing.assert_array_equal(env.observer_view(a).features, env.observer_view(b).features)
    assert np.any(env.observer_view(a).features[-(config.readers + 2):])


def test_legitimate_actor_has_no_access_to_private_observer_knowledge(config, dataset):
    env = SemanticEnv(config, dataset)
    obs = env.reset(19)
    env.eve_theta[:] = 1 - env.eve_theta
    env.eve_chi[:] = ~env.eve_chi
    env.budget = 0
    after = env.legitimate_view()
    np.testing.assert_array_equal(obs.items, after.items)
    np.testing.assert_array_equal(obs.global_features, after.global_features)
    np.testing.assert_array_equal(obs.history, after.history)


def test_passive_observer_cannot_change_legitimate_transitions(config, dataset):
    a, b = SemanticEnv(config, dataset), SemanticEnv(config, dataset)
    a.reset(23)
    b.reset(23)
    assert a.pair_fingerprint == b.pair_fingerprint
    while not a.done:
        assert a.legal_actions() == b.legal_actions()
        action = a.legal_actions()[-1]
        a.step(action, False)
        b.step(action, b.budget > 0)
        np.testing.assert_array_equal(a.theta, b.theta)
        np.testing.assert_array_equal(a.completed, b.completed)
    assert b.c.observer_budget - b.budget <= config.observer_budget


def test_kfb_consumes_action_but_never_completes_task(config, dataset):
    env = SemanticEnv(config, dataset)
    env.reset(29)
    env.theta[:] = 0
    env.chi[:] = False
    env.step(env.legal_actions()[0])
    feedback = [a for a in env.legal_actions() if a.kind == 1][0]
    before, t = env.completed.copy(), env.t
    result = env.step(feedback)
    np.testing.assert_array_equal(env.completed, before)
    assert env.t == t + 1 and result['mu'] == 0


def test_repair_buffer_overflow_preserves_waiting_tasks(config, dataset):
    env = SemanticEnv(replace(config, feedback_capacity=1), dataset)
    env.reset(31)
    env.chi[:] = False
    env.theta[:] = 0
    data = [a for a in env.legal_actions() if a.kind == 0]
    env.step(data[0])
    env.step(data[1])
    assert len(env.repairs) == 1 and len(env.waiting) == 1
    assert all(a.task != data[1].task for a in env.legal_actions())
    assert env.completed.sum() + (~env.completed).sum() == config.tasks
    env.step(next(a for a in env.legal_actions() if a.kind == 1))
    assert not env.waiting and len(env.repairs) == 1


def test_unobserved_data_and_status_do_not_enter_e0_history(config, dataset):
    env = SemanticEnv(config, dataset)
    env.reset(37)
    env.step(env.legal_actions()[0], False)
    assert not np.any(env.observer_view(env.legal_actions()[0]).history)

