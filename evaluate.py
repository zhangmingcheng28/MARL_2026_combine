from arguments import build_parser
from config.loader import load_config, merge_cli_into_config
from envs.shared_env import SharedMultiAgentEnv
from agents import build_trainer
from runners.eval_runner import EvalRunner


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_into_config(config, args)

    env = SharedMultiAgentEnv(
        n_agents=config["env"]["n_agents"],
        obs_dim=config["env"]["obs_dim"],
        action_dim=config["env"]["action_dim"],
        max_steps=config["env"]["max_steps"],
    )

    trainer = build_trainer(config)
    checkpoint = args.checkpoint or f"checkpoints/{args.exp_name}"
    trainer.load(checkpoint)

    runner = EvalRunner(env=env, trainer=trainer, config=config)
    runner.run()


if __name__ == "__main__":
    main()
