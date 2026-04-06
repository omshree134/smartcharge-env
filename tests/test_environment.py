from env.agent import edf_policy
from env.environment import FAST_RATE, SLOW_RATE, SmartChargeEnv
from env.models import Action, Vehicle


def test_reset_returns_mode_specific_observation():
    env = SmartChargeEnv(mode="medium", seed=42)
    obs = env.reset()

    assert obs.time_step == 0
    assert obs.slots_available == 3
    assert obs.vehicles == []
    assert 0.1 <= obs.price <= 1.0
    assert 0.0 <= obs.renewable <= 1.0


def test_charging_updates_soc_with_renewable_scaling():
    env = SmartChargeEnv(mode="easy", seed=7)
    env.reset()
    env.renewable = 0.5
    env.vehicles = [Vehicle(id="veh-1", soc=0.2, deadline=3, priority="normal")]

    obs, reward, done, info = env.step(Action(assignments=[1]))

    expected_soc = min(1.0, 0.2 + SLOW_RATE * (1.0 + 0.5 * 0.2))
    assert obs.time_step == 1
    assert reward >= 0.0
    assert done is False
    assert info["power_used"] == 1.0
    assert env.vehicles[0].soc == expected_soc


def test_served_and_missed_are_tracked():
    env = SmartChargeEnv(mode="easy", seed=1)
    env.reset()
    env.renewable = 0.0
    env.vehicles = [
        Vehicle(id="full", soc=0.95, deadline=1, priority="high"),
        Vehicle(id="late", soc=0.2, deadline=1, priority="normal"),
    ]

    _, _, _, info = env.step(Action(assignments=[2, 0]))

    assert info["served_on_time"] == 1
    assert info["missed"] == 1


def test_reward_is_bounded_and_penalizes_spikes():
    env = SmartChargeEnv(mode="medium", seed=4)
    env.reset()
    env.vehicles = [
        Vehicle(id="veh-1", soc=0.1, deadline=5, priority="high"),
        Vehicle(id="veh-2", soc=0.2, deadline=5, priority="normal"),
        Vehicle(id="veh-3", soc=0.3, deadline=5, priority="normal"),
    ]

    _, reward_a, _, info_a = env.step(Action(assignments=[2, 2, 2]))
    env.vehicles = [
        Vehicle(id="veh-4", soc=0.1, deadline=5, priority="high"),
        Vehicle(id="veh-5", soc=0.2, deadline=5, priority="normal"),
        Vehicle(id="veh-6", soc=0.3, deadline=5, priority="normal"),
    ]
    _, reward_b, _, info_b = env.step(Action(assignments=[2, 2, 2]))

    assert 0.0 <= reward_a <= 1.0
    assert 0.0 <= reward_b <= 1.0
    assert info_a["reward_breakdown"]["energy_efficiency"] <= 1.0
    assert info_b["reward_breakdown"]["energy_efficiency"] <= info_a["reward_breakdown"]["energy_efficiency"]


def test_policy_respects_slots_and_price_awareness():
    env = SmartChargeEnv(mode="easy", seed=3)
    env.reset()
    env.price = 0.9
    env.vehicles = [
        Vehicle(id="urgent", soc=0.1, deadline=1, priority="high"),
        Vehicle(id="flex", soc=0.7, deadline=10, priority="normal"),
        Vehicle(id="soon", soc=0.3, deadline=2, priority="normal"),
    ]

    action = edf_policy(env.state())

    assert len(action.assignments) == 3
    assert sum(1 for value in action.assignments if value > 0) <= env.max_slots
    assert action.assignments[0] == 2
    assert action.assignments[1] == 0


def test_same_seed_produces_same_trajectory():
    env_a = SmartChargeEnv(mode="hard", seed=99)
    env_b = SmartChargeEnv(mode="hard", seed=99)

    obs_a = env_a.reset()
    obs_b = env_b.reset()
    assert obs_a == obs_b

    for _ in range(5):
        action_a = edf_policy(obs_a)
        action_b = edf_policy(obs_b)
        obs_a, reward_a, done_a, info_a = env_a.step(action_a)
        obs_b, reward_b, done_b, info_b = env_b.step(action_b)
        assert obs_a == obs_b
        assert reward_a == reward_b
        assert done_a == done_b
        assert info_a == info_b
