from agents.iddpg.trainer import IDDPGTrainer
from agents.maddpg.trainer import MADDPGTrainer
from agents.matd3.trainer import MATD3Trainer


def build_trainer(config):
    algo = config["algorithm"].lower()

    if algo == "iddpg":
        return IDDPGTrainer(config)
    if algo == "maddpg":
        return MADDPGTrainer(config)
    if algo == "matd3":
        return MATD3Trainer(config)

    raise ValueError(f"Unknown algorithm: {algo}")
