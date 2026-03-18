from pathlib import Path
from utils.logger import Logger


class TrainRunner:
    def __init__(self, env, trainer, config):
        self.env = env
        self.trainer = trainer
        self.config = config

    def run(self):
        num_episodes = self.config["train"]["num_episodes"]
        max_steps = self.config["env"]["max_steps"]
        ckpt_dir = Path("checkpoints") / self.config["exp_name"]
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for episode in range(1, num_episodes + 1):
            obs = self.env.reset()
            total_reward = 0.0

            for _ in range(max_steps):
                actions = self.trainer.select_action(obs, evaluate=False)
                next_obs, rewards, dones, info = self.env.step(actions)

                self.trainer.store_transition(obs, actions, rewards, next_obs, dones)
                self.trainer.update()

                total_reward += sum(rewards)
                obs = next_obs

                if all(dones):
                    break

            Logger.log_episode(episode, total_reward)

        self.trainer.save(str(ckpt_dir))
