import matplotlib.pyplot as plt
from pettingzoo.utils.env import AECEnv

from agent import Agent


def play_game(env: AECEnv, agent0: Agent, agent1: Agent):
    env.reset()
    obs, _, done, _, _ = env.last()
    player, other = agent0, agent1
    nb_rounds = 0
    while not done:
        nb_rounds += 1
        action = player.get_action(env)
        env.step(action)
        obs, reward, done, _, _ = env.last()
        player, other = other, player
    if reward == 0:
        return "draw", nb_rounds
    return other.name, nb_rounds
