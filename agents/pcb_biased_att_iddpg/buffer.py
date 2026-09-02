from collections import namedtuple
import random


Experience = namedtuple(
    "Experience",
    (
        "states_obs",
        "states_grid",
        "states_var_nei",
        "actions",
        "next_states_obs",
        "next_states_grid",
        "next_states_var_nei",
        "rewards",
        "dones",
    ),
)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(
        self,
        states_obs,
        states_grid,
        states_var_nei,
        actions,
        next_states_obs,
        next_states_grid,
        next_states_var_nei,
        rewards,
        dones,
    ):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Experience(
            states_obs,
            states_grid,
            states_var_nei,
            actions,
            next_states_obs,
            next_states_grid,
            next_states_var_nei,
            rewards,
            dones,
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
