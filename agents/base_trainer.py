from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def select_action(self, obs, evaluate=False):
        raise NotImplementedError

    @abstractmethod
    def store_transition(self, obs, actions, rewards, next_obs, dones):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str, episode: int = None, step: int = None):
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str, checkpoint_tag: str = None):
        raise NotImplementedError
