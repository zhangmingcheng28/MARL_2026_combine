class Logger:
    @staticmethod
    def log_episode(episode, reward):
        print(f"Episode {episode:04d} | Total Reward = {reward:.3f}")
