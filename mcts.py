import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from typing import List, Optional, Tuple

from pettingzoo.utils.env import AECEnv

from agent import Agent
from utils import copy_connect4_env


class State:
    def __init__(self, previous_env: AECEnv, last_player: int):
        self.env = copy_connect4_env(previous_env, last_player)

        self.update_last()
        self.last_player = last_player

    def update_last(self):
        last_step = self.env.last()
        self.state: np.ndarray = last_step[0]["observation"]
        self.game_result: int = last_step[1]
        self.is_game_over: bool = last_step[2]
        self.legal_actions: List[int] = np.where(last_step[0]["action_mask"])[
            0
        ].tolist()

    def move(self, action: int):
        self.env.step(action)
        self.last_player = 1 - self.last_player
        self.update_last()


class MonteCarloTreeSearchNode:
    def __init__(
        self,
        state: State,
        parent: Optional["MonteCarloTreeSearchNode"] = None,
        parent_action=None,
        number=0,
        n_rollout=3,
    ) -> None:
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children: dict[int, MonteCarloTreeSearchNode] = {}
        self.number = number
        self.n_rollout = n_rollout
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = [action for action in self.state.legal_actions]

    @property
    def q(self) -> int:
        wins = self._results[1]
        looses = self._results[-1]
        return wins - looses

    @property
    def n(self) -> int:
        return self._number_of_visits

    @property
    def is_terminal_node(self) -> bool:
        return self.state.is_game_over

    @property
    def is_fully_expanded(self) -> bool:
        return len(self._untried_actions) == 0

    def add_child_node(self, action: int) -> "MonteCarloTreeSearchNode":
        next_state = State(self.state.env, last_player=self.state.last_player)
        next_state.move(action)
        child_node = MonteCarloTreeSearchNode(
            next_state, self, action, number=self.number, n_rollout=self.n_rollout
        )
        self.children[action] = child_node
        return child_node

    def expand(self) -> "MonteCarloTreeSearchNode":
        action = self._untried_actions.pop()
        child_node = self.add_child_node(action)
        return child_node

    def rollout_policy(self, possible_moves):
        return np.random.choice(possible_moves)

    def rollout(self) -> Tuple[int, int]:
        rollout_env = copy_connect4_env(self.state.env, self.state.last_player)

        last_player = self.state.last_player
        state, reward, terminated, _, _ = rollout_env.last()
        i = 0
        distance = 0
        while not terminated:
            distance += 1
            possible_moves = np.where(state["action_mask"])[0].tolist()
            action = self.rollout_policy(possible_moves)
            rollout_env.step(action)
            state, reward, terminated, _, _ = rollout_env.last()
            last_player = 1 - last_player
            i += 1

        if reward == 0:  # We consider draw as a defeat
            return -1, distance

        return (
            (4 * last_player * self.number - 2 * last_player - 2 * self.number + 1),
            distance,
        )

    def backpropagate(self, result: int, distance: int):
        self._number_of_visits += 1
        if distance > 0:
            self._results[result] += 1 / distance
        else:
            self._results[result] += 2

        if self.parent:
            self.parent.backpropagate(result, distance + 1)

    def get_action_weight(self, action: int, c_param=np.sqrt(2)):
        return self.children[action].q / self.children[action].n + c_param * np.sqrt(
            2 * np.log(self.n) / self.children[action].n
        )

    def best_child(self, c_param=np.sqrt(2)) -> int:
        best_action = max(
            self.state.legal_actions,
            key=lambda action: self.get_action_weight(action, c_param),
        )
        return best_action

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node:
            if not current_node.is_fully_expanded:
                return current_node.expand()

            else:
                current_node = current_node.children[current_node.best_child()]

        return current_node

    def best_action(self):
        for _ in range(self.n_rollout):
            v = self._tree_policy()
            reward, distance = v.rollout()
            v.backpropagate(reward, distance)
        return self.best_child(c_param=0)

    def get_children_from_state(self, action: int) -> "MonteCarloTreeSearchNode":
        return self.children[action]


class MCTSPlayer(Agent):
    def __init__(self, name="MCTS Player", number=0, n_rollout=500) -> None:
        super().__init__(name, number)
        self.root: Optional[MonteCarloTreeSearchNode] = None
        self.n_rollout = n_rollout
        self.number = number

    def update_root(self, action):
        if action not in self.root.children:
            self.root.add_child_node(action)
        self.root = self.root.children[action]

    def get_action_from_state(self, new_state: np.ndarray):
        other_play = new_state[:, :, 1]
        previous_other_play = self.root.state.state[:, :, 0]
        return np.where(other_play - previous_other_play)[1][0]

    def get_action(self, env: AECEnv) -> int:
        if self.root is not None:
            state, _, _, _, _ = env.last()
            other_action = self.get_action_from_state(state["observation"])
            self.update_root(other_action)
        else:
            actual_state = State(env, last_player=1 - self.number)
            self.root = MonteCarloTreeSearchNode(
                state=actual_state, n_rollout=self.n_rollout, number=self.number
            )
        best_action = self.root.best_action()
        self.update_root(best_action)
        return best_action
