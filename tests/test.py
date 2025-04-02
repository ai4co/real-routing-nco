import pytest
from rl4co.envs import get_env

from rrnco.models import RRNetPolicy


@pytest.mark.parametrize("env_name", ["rcvrp", "rcvrptw", "atsp"])
def test_rrnco(env_name):
    # Create environment
    env = get_env(env_name, generator_params={"num_loc": 20})

    # Generate test data
    td_test_data = env.generator(batch_size=[2])
    td_init = env.reset(td_test_data.clone())
    td_init_test = td_init.clone()

    # Create RRNCO policy
    policy = RRNetPolicy(env_name=env_name)

    # Execute policy
    out = policy(td_init_test.clone(), env)

    # Verify outputs
    assert out["reward"].shape == (2,)
    assert out["actions"].shape[0] == 2  # batch size
    assert out["log_likelihood"].shape == (2,)
