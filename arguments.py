import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="MARL training/evaluation skeleton")
    parser.add_argument("--algo", type=str, default="iddpg", choices=["iddpg", "maddpg", "matd3"])
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--exp-name", type=str, default="default_exp")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-gru", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser
