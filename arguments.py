import argparse
from copy import deepcopy

from config.paths import DEFAULT_RESOURCE_ENV_VAR, PROJECT_ROOT, resolve_path


DEFAULT_CONFIG = {
    "seed": 777,
    "device": "auto",
    "dtype": "float32",
    "mode": "train",  # or evaluate
    "algorithm": "iddpg",
    "exp_name": "default_exp",
    "save_interval": 10,
    "paths": {
        "project_root": str(PROJECT_ROOT),
        "resource_env_var": DEFAULT_RESOURCE_ENV_VAR,
        "checkpoint_dir": "checkpoints",
        "resource_file": None,
        "shape_file": "resources/lakesideMap/lakeSide.shp",
        "agent_config_file": "resources/fixedDrone_3drones.xlsx",
        "legacy_code_dir": None,
    },
    "env": {
        "n_agents": 8,
        "obs_dim": 6,
        "action_dim": 2,
        "max_steps": 100,
        "grid_obs_shape": [7, 7],
        "bound": [455, 680, 255, 385],
        "max_x": 1800,
        "max_y": 1300,
        "grid_length": 10,
        "acc_max": 8,
        "max_speed": 5,
        "neighbour_search_distance": 10000,
        "full_observable_critic": False,
        "evaluation_by_episode": False,
    },
    "train": {
        "num_episodes": 20000,
        "total_steps": 2000000,
        "stop_mode": "episode",  # or "step"
        "batch_size": 256,
        "buffer_size": 1000000,
        "actor_lr": 0.0001,
        "critic_lr": 0.0001,
        "gamma": 0.95,
        "tau": 0.01,
        "hidden_dim": 128,
        "update_every": 1,
    },
    "exploration": {
        "eps_start": 1.0,
        "eps_end": 0.03,
        "eps_period": 8500,  # the noise finishes decaying and reaches eps_end.”
        "largest_noise_sigma": 0.5,
        "smallest_noise_sigma": 0.15,
        "initial_noise_sigma": 0.5,
    },
    "flags": {
        "use_wandb": False,
        "evaluation_by_episode": False,
        "get_evaluation_status": False,
        "simply_view_evaluation": False,
        "full_observable_critic": False,
        "transfer_learning": False,
        "use_gru": False,
        "use_single_portion_selfatt": False,
        "use_selfatt_with_radar": False,
        "use_all_neigh_with_radar": True,
        "own_obs_only": False,

    },
    "eval": {
        "episodes": 3,
    },
}


def _parse_int_list(raw_value):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def _resolve_optional_path(raw_path):
    if raw_path in (None, ""):
        return None
    return resolve_path(raw_path)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default=DEFAULT_CONFIG["mode"], choices=["train", "evaluate"])
    parser.add_argument("--algo", type=str, default=DEFAULT_CONFIG["algorithm"], choices=["iddpg", "maddpg", "matd3"])
    parser.add_argument("--exp_name", type=str, default=DEFAULT_CONFIG["exp_name"])

    parser.add_argument("--n_agents", type=int, default=DEFAULT_CONFIG["env"]["n_agents"])
    parser.add_argument("--obs_dim", type=int, default=DEFAULT_CONFIG["env"]["obs_dim"])
    parser.add_argument("--action_dim", type=int, default=DEFAULT_CONFIG["env"]["action_dim"])
    parser.add_argument("--max_steps", type=int, default=DEFAULT_CONFIG["env"]["max_steps"])
    parser.add_argument("--max_x", type=int, default=DEFAULT_CONFIG["env"]["max_x"])
    parser.add_argument("--max_y", type=int, default=DEFAULT_CONFIG["env"]["max_y"])
    parser.add_argument("--grid_length", type=int, default=DEFAULT_CONFIG["env"]["grid_length"])
    parser.add_argument("--grid_obs_shape", type=_parse_int_list, default=list(DEFAULT_CONFIG["env"]["grid_obs_shape"]))
    parser.add_argument("--bound", type=_parse_int_list, default=list(DEFAULT_CONFIG["env"]["bound"]))
    parser.add_argument("--acc_max", type=float, default=DEFAULT_CONFIG["env"]["acc_max"])
    parser.add_argument("--max_speed", type=float, default=DEFAULT_CONFIG["env"]["max_speed"])

    parser.add_argument("--num_episodes", type=int, default=DEFAULT_CONFIG["train"]["num_episodes"])
    parser.add_argument("--total_steps", type=int, default=DEFAULT_CONFIG["train"]["total_steps"])
    parser.add_argument(
        "--stop_mode",
        type=str,
        default=DEFAULT_CONFIG["train"]["stop_mode"],
        choices=["episode", "step"],
    )
    parser.add_argument("--eval_episodes", type=int, default=DEFAULT_CONFIG["eval"]["episodes"])
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_CONFIG["train"]["hidden_dim"])
    parser.add_argument("--actor_lr", type=float, default=DEFAULT_CONFIG["train"]["actor_lr"])
    parser.add_argument("--critic_lr", type=float, default=DEFAULT_CONFIG["train"]["critic_lr"])
    parser.add_argument("--gamma", type=float, default=DEFAULT_CONFIG["train"]["gamma"])
    parser.add_argument("--tau", type=float, default=DEFAULT_CONFIG["train"]["tau"])
    parser.add_argument("--buffer_size", type=int, default=DEFAULT_CONFIG["train"]["buffer_size"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["train"]["batch_size"])
    parser.add_argument("--update_every", type=int, default=DEFAULT_CONFIG["train"]["update_every"])

    parser.add_argument("--eps_start", type=float, default=DEFAULT_CONFIG["exploration"]["eps_start"])
    parser.add_argument("--eps_end", type=float, default=DEFAULT_CONFIG["exploration"]["eps_end"])
    parser.add_argument("--eps_period", type=int, default=DEFAULT_CONFIG["exploration"]["eps_period"])
    parser.add_argument("--largest_noise_sigma", type=float, default=DEFAULT_CONFIG["exploration"]["largest_noise_sigma"])
    parser.add_argument("--smallest_noise_sigma", type=float, default=DEFAULT_CONFIG["exploration"]["smallest_noise_sigma"])
    parser.add_argument("--initial_noise_sigma", type=float, default=DEFAULT_CONFIG["exploration"]["initial_noise_sigma"])

    parser.add_argument("--use_wandb", action="store_true", default=DEFAULT_CONFIG["flags"]["use_wandb"])
    parser.add_argument("--evaluation_by_episode", action="store_true", default=DEFAULT_CONFIG["flags"]["evaluation_by_episode"])
    parser.add_argument("--get_evaluation_status", action="store_true", default=DEFAULT_CONFIG["flags"]["get_evaluation_status"])
    parser.add_argument("--simply_view_evaluation", action="store_true", default=DEFAULT_CONFIG["flags"]["simply_view_evaluation"])
    parser.add_argument("--full_observable_critic", action="store_true", default=DEFAULT_CONFIG["flags"]["full_observable_critic"])
    parser.add_argument("--transfer_learning", action="store_true", default=DEFAULT_CONFIG["flags"]["transfer_learning"])
    parser.add_argument("--use_gru", action="store_true", default=DEFAULT_CONFIG["flags"]["use_gru"])
    parser.add_argument("--use_single_portion_selfatt", action="store_true", default=DEFAULT_CONFIG["flags"]["use_single_portion_selfatt"])
    parser.add_argument("--use_selfatt_with_radar", action="store_true", default=DEFAULT_CONFIG["flags"]["use_selfatt_with_radar"])
    parser.add_argument("--use_all_neigh_with_radar", action="store_true", default=DEFAULT_CONFIG["flags"]["use_all_neigh_with_radar"])
    parser.add_argument("--own_obs_only", action="store_true", default=DEFAULT_CONFIG["flags"]["own_obs_only"])

    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"])
    parser.add_argument("--dtype", type=str, default=DEFAULT_CONFIG["dtype"], choices=["float32", "float64"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--save_interval", type=int, default=DEFAULT_CONFIG["save_interval"])

    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CONFIG["paths"]["checkpoint_dir"])
    parser.add_argument("--resource_file", type=str, default=DEFAULT_CONFIG["paths"]["resource_file"])
    parser.add_argument("--shape_file", type=str, default=DEFAULT_CONFIG["paths"]["shape_file"])
    parser.add_argument("--agent_config_file", type=str, default=DEFAULT_CONFIG["paths"]["agent_config_file"])
    parser.add_argument("--legacy_code_dir", type=str, default=DEFAULT_CONFIG["paths"]["legacy_code_dir"])

    return parser.parse_args()


def build_config(args):
    config = deepcopy(DEFAULT_CONFIG)

    config["mode"] = args.mode
    config["algorithm"] = args.algo
    config["exp_name"] = args.exp_name
    config["seed"] = args.seed
    config["device"] = args.device
    config["dtype"] = args.dtype
    config["save_interval"] = args.save_interval

    config["paths"]["checkpoint_dir"] = resolve_path(args.checkpoint_dir)
    config["paths"]["resource_file"] = _resolve_optional_path(args.resource_file)
    config["paths"]["shape_file"] = resolve_path(args.shape_file)
    config["paths"]["agent_config_file"] = resolve_path(args.agent_config_file)
    config["paths"]["legacy_code_dir"] = _resolve_optional_path(args.legacy_code_dir)

    config["env"]["n_agents"] = args.n_agents
    config["env"]["obs_dim"] = args.obs_dim
    config["env"]["action_dim"] = args.action_dim
    config["env"]["max_steps"] = args.max_steps
    config["env"]["grid_obs_shape"] = list(args.grid_obs_shape)
    config["env"]["bound"] = list(args.bound)
    config["env"]["acc_max"] = args.acc_max
    config["env"]["max_speed"] = args.max_speed
    config["env"]["resource_file"] = config["paths"]["resource_file"]

    config["train"]["num_episodes"] = args.num_episodes
    config["train"]["total_steps"] = args.total_steps
    config["train"]["stop_mode"] = args.stop_mode
    config["train"]["batch_size"] = args.batch_size
    config["train"]["buffer_size"] = args.buffer_size
    config["train"]["actor_lr"] = args.actor_lr
    config["train"]["critic_lr"] = args.critic_lr
    config["train"]["gamma"] = args.gamma
    config["train"]["tau"] = args.tau
    config["train"]["hidden_dim"] = args.hidden_dim
    config["train"]["update_every"] = args.update_every

    config["exploration"]["eps_start"] = args.eps_start
    config["exploration"]["eps_end"] = args.eps_end
    config["exploration"]["eps_period"] = args.eps_period
    config["exploration"]["largest_noise_sigma"] = args.largest_noise_sigma
    config["exploration"]["smallest_noise_sigma"] = args.smallest_noise_sigma
    config["exploration"]["initial_noise_sigma"] = args.initial_noise_sigma

    config["flags"]["use_wandb"] = args.use_wandb
    config["flags"]["evaluation_by_episode"] = args.evaluation_by_episode
    config["flags"]["get_evaluation_status"] = args.get_evaluation_status
    config["flags"]["simply_view_evaluation"] = args.simply_view_evaluation
    config["flags"]["full_observable_critic"] = args.full_observable_critic
    config["flags"]["transfer_learning"] = args.transfer_learning
    config["flags"]["use_gru"] = args.use_gru
    config["flags"]["use_single_portion_selfatt"] = args.use_single_portion_selfatt
    config["flags"]["use_selfatt_with_radar"] = args.use_selfatt_with_radar
    config["flags"]["use_all_neigh_with_radar"] = args.use_all_neigh_with_radar
    config["flags"]["own_obs_only"] = args.own_obs_only

    config["env"]["full_observable_critic"] = config["flags"]["full_observable_critic"]
    config["env"]["evaluation_by_episode"] = config["flags"]["evaluation_by_episode"]

    config["eval"]["episodes"] = args.eval_episodes
    return config
