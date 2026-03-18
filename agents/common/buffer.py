import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, obs, actions, rewards, next_obs, dones):
        transition = (obs, actions, rewards, next_obs, dones)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = map(list, zip(*batch))
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)
