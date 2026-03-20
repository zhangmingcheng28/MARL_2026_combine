from collections import namedtuple
import random


Experience = namedtuple(
    "Experience",
    (
        "states_obs",
        "states_nei",
        "states_grid",
        "actions",
        "next_states_obs",
        "next_states_nei",
        "next_states_grid",
        "rewards",
        "dones",
        "history_info",
        "cur_hidden",
        "next_hidden",
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
        states_nei,
        states_grid,
        actions,
        next_states_obs,
        next_states_nei,
        next_states_grid,
        rewards,
        dones,
        history_info=None,
        cur_hidden=None,
        next_hidden=None,
    ):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Experience(
            states_obs,
            states_nei,
            states_grid,
            actions,
            next_states_obs,
            next_states_nei,
            next_states_grid,
            rewards,
            dones,
            history_info,
            cur_hidden,
            next_hidden,
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
