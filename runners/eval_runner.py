import numpy as np


class EvalRunner:
    def __init__(self, env, trainer, config):
        self.env = env
        self.trainer = trainer
        self.config = config

    def run(self):
        episode_rewards = []
        episodes = self.config["eval"]["episodes"]
        max_steps = self.config["env"]["max_steps"]

        for ep in range(1, episodes + 1):
            obs = self.env.reset()
            total_reward = 0.0

            for _ in range(max_steps):
                actions = self.trainer.select_action(obs, evaluate=True)
                next_obs, rewards, dones, info = self.env.step(actions)

                total_reward += sum(rewards)
                obs = next_obs

                if all(dones):
                    break

            episode_rewards.append(total_reward)
            print(f"Eval Episode {ep:04d} | Total Reward = {total_reward:.3f}")

        print(f"Mean Eval Reward = {np.mean(episode_rewards):.3f}")
