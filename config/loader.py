from copy import deepcopy
import yaml


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_cli_into_config(config: dict, args):
    cfg = deepcopy(config)
    cfg["algorithm"] = args.algo
    cfg["exp_name"] = args.exp_name
    cfg["model"] = cfg.get("model", {})
    cfg["model"]["use_gru"] = bool(args.use_gru)

    if args.device is not None:
        cfg["device"] = args.device

    return cfg
