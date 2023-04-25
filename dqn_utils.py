import random
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
from copy import deepcopy


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, terminated, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, terminated, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    """
    Basic neural net.
    """

    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQN_Skeleton:
    def __init__(
        self,
        action_space_size,
        observation_space_size,
        gamma,
        batch_size,
        buffer_capacity,
        update_target_every,
        epsilon_start,
        decrease_epsilon_factor,
        epsilon_min,
        learning_rate,
        env,
    ):
        self.action_space_size = action_space_size
        self.observation_space_size = observation_space_size
        self.gamma = gamma

        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.update_target_every = update_target_every

        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = (
            decrease_epsilon_factor  # larger -> more exploration
        )
        self.epsilon_min = epsilon_min

        self.learning_rate = learning_rate

        self.env = env

        self.name = "DQN Agent"

        self.reset()

    def get_action(self, state, epsilon=None):
        """
        ** Solution **

        Return action according to an epsilon-greedy exploration policy
        """
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:
            playable_moves = np.where(state["action_mask"] == 1)[0]
            return np.random.choice(playable_moves)
        else:
            q_prediction = self.get_q(state)
            # set unplayable moves to -∞ to avoid playing them
            q_prediction[state["action_mask"] == 0] = -np.inf
            return np.argmax(q_prediction)

    def update(self, state, action, reward, terminated, next_state):
        """
        ** SOLUTION **
        """
        if state is None or next_state is None:
            return None

        state_tensor = self.state_observation_to_DQN_input(state["observation"])
        next_state_tensor = self.state_observation_to_DQN_input(state["observation"])

        # add data to replay buffer
        self.buffer.push(
            torch.tensor(state_tensor).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.int64),
            torch.tensor([reward]),
            torch.tensor([terminated], dtype=torch.int64),
            torch.tensor(next_state_tensor).unsqueeze(0),
        )

        if len(self.buffer) < self.batch_size:
            return None

        # get batch
        transitions = self.buffer.sample(self.batch_size)

        (
            state_batch,
            action_batch,
            reward_batch,
            terminated_batch,
            next_state_batch,
        ) = tuple([torch.cat(data) for data in zip(*transitions)])

        values = self.q_net.forward(state_batch).gather(1, action_batch)

        # Compute the ideal Q values
        with torch.no_grad():
            next_state_values = (1 - terminated_batch) * self.target_net(
                next_state_batch
            ).max(1)[0]
            targets = next_state_values * self.gamma + reward_batch

        loss = self.loss_function(values, targets.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        self.optimizer.step()

        if not ((self.n_steps + 1) % self.update_target_every):
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.decrease_epsilon()

        self.n_steps += 1
        if terminated:
            self.n_eps += 1

        return loss.detach().numpy()

    def get_q(self, state):
        """
        Compute Q function for a states
        """
        state_tensor = self.state_observation_to_DQN_input(
            state["observation"]
        ).unsqueeze(0)

        with torch.no_grad():
            output = self.q_net.forward(state_tensor)  # shape (1,  n_actions)
        return output.numpy()[0]  # shape  (n_actions)

    def decrease_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
            np.exp(-1.0 * self.n_eps / self.decrease_epsilon_factor)
        )

    def state_observation_to_DQN_input(
        self, state_observation, my_value=1, ennemy_value=2
    ):
        my_pieces = 0
        adverse_pieces = 1
        result = np.zeros(state_observation.shape[:2])
        result[np.where(state_observation[:, :, my_pieces] == 1)] = my_value
        result[np.where(state_observation[:, :, adverse_pieces] == 1)] = ennemy_value
        return torch.from_numpy(result.flatten()).float()

    def reset(self):
        hidden_size = 128

        obs_size = self.observation_space_size
        n_actions = self.action_space_size

        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.q_net = Net(obs_size, hidden_size, n_actions)
        self.target_net = Net(obs_size, hidden_size, n_actions)

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(
            params=self.q_net.parameters(), lr=self.learning_rate
        )

        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0


class RandomAgent:
    def __init__(self):
        self.name = "Random Agent"

    def get_action(self, state, *args, **kwargs):
        playable_moves = np.where(state["action_mask"] == 1)[0]
        return np.random.choice(playable_moves)

    def update(self, *data):
        pass


class PlayLeftmostLegal:
    def __init__(self):
        self.name = "Left Player"

    def get_action(self, obs_mask, epsilon=None):
        for i, legal in enumerate(obs_mask["action_mask"]):
            if legal:
                return i
        return None
