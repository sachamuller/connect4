import numpy as np
from copy import copy

from pettingzoo.utils.env import AECEnv
from pettingzoo.classic import connect_four_v3


def copy_connect4_env(previous_env: AECEnv, last_player: int) -> AECEnv:
    env = connect_four_v3.env(render_mode="rgb_array")
    env.reset()
    env.env.env.env.board = np.copy(previous_env.env.env.env.board)
    env.terminations = copy(previous_env.terminations)
    env.rewards = copy(previous_env.rewards)
    env._cumulative_rewards = copy(previous_env._cumulative_rewards)
    env.last()
    if env.env.env.env.agent_selection == f"player_{last_player}":
        env.env.env.env.agent_selection = env.env.env.env._agent_selector.next()
        env.agent_selection = env.env.env.env.agent_selection
    return env
