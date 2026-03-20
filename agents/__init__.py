from agents.iddpg.trainer import IDDPGTrainer
from agents.maddpg.trainer import MADDPGTrainer
from agents.matd3.trainer import MATD3Trainer


def build_trainer(args):
    algo = args["algorithm"].lower()

    if algo == "iddpg":
        return IDDPGTrainer(args)
    elif algo == "maddpg":
        return MADDPGTrainer(args)
    elif algo == "matd3":
        return MATD3Trainer(args)
    else:
        raise ValueError(f"Unknown algorithm: {args['algorithm']}")
