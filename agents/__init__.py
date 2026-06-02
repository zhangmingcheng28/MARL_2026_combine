from agents.fm_iddpg.trainer import FMIDDPGTrainer
from agents.iddpg.trainer import IDDPGTrainer
from agents.maac.trainer import MAACTrainer
from agents.maddpg.trainer import MADDPGTrainer
from agents.mappo.trainer import MAPPOTrainer
from agents.matd3.trainer import MATD3Trainer
from agents.orca.trainer import ORCATrainer


def build_trainer(args):
    algo = args["algorithm"].lower()

    if algo == "iddpg":
        return IDDPGTrainer(args)
    elif algo == "fm-iddpg":
        return FMIDDPGTrainer(args)
    elif algo in ("maddpg", "maddpg-critic-attention"):
        return MADDPGTrainer(args)
    elif algo == "mappo":
        return MAPPOTrainer(args)
    elif algo == "maac":
        return MAACTrainer(args)
    elif algo in ("matd3", "matd3-critic-attention"):
        return MATD3Trainer(args)
    elif algo == "orca":
        return ORCATrainer(args)
    else:
        raise ValueError(f"Unknown algorithm: {args['algorithm']}")
