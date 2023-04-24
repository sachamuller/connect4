import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.utils.env import AECEnv

from utils import copy_connect4_env


class Agent:
    def __init__(self, name: str, number: int) -> None:
        self.name = name
        self.number = number

    def get_action(self, env: AECEnv) -> int:
        raise Exception("You must implement a 'get_action' method")


class RandomPlayer(Agent):
    def __init__(self, name="Random Player", number=1) -> None:
        super().__init__(name, number)

    def get_action(self, env: AECEnv) -> int:
        env = copy_connect4_env(env, 1 - self.number)
        obs, _, _, _, _ = env.last()
        return np.random.choice(np.where(obs["action_mask"])[0])


class UserPlayer(Agent):
    def __init__(self, name="User Player", number=0) -> None:
        super().__init__(name, number)

    def get_action(self, env: AECEnv) -> int:
        obs, _, _, _, _ = env.last()
        possible_moves = list(np.where(obs["action_mask"])[0])
        move = -1
        plt.imshow(env.render())
        plt.show()
        while int(move) not in possible_moves:
            move = input(f"Choose a move among those ones: {possible_moves}")
        return int(move)
