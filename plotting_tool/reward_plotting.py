# -*- coding: utf-8 -*-
r"""
Read training pickles from one or more run ``toplot`` directories and plot rewards.

Examples:
    py -3 plotting_tool/reward_plotting.py 020626_05_47_51
    py -3 plotting_tool/reward_plotting.py F:\githubClone\MARL_2026_combine\checkpoints\default_exp\020626_05_47_51\toplot
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP_NAME = "default_exp"

# Quick-look input. This can be either:
#   1. a run id, for example "020626_05_47_51"
#   2. a full run directory
#   3. a full toplot directory
TOPLOT_INPUT = r"F:\githubClone\MARL_2026_combine\checkpoints\default_exp\200726_08_49_19\toplot"

# Reward comparison inputs. Set USE_REWARD_COMPARISON = False to use TOPLOT_INPUT
# for a single-run plot.
USE_REWARD_COMPARISON = True
PCB_BIASED_ATT_IDDPG_RUN = "200726_08_49_19"
MADDPG_RUN = "070426_07_24_34"
MAAC_RUN = "210526_06_41_16"
IDDPG_RUN = "070426_06_56_48"
MAPPO_RUN = "020626_05_51_16"
MATD3_RUN = "190526_09_31_31"

EXP_NAME = DEFAULT_EXP_NAME
MOVING_AVERAGE_WINDOW = 100
SHOW_PLOT = True
OUTPUT_PATH = None  # Single-run: <toplot>/reward_curve.png; multi-run: plotting_tool/reward_comparison.png
PLOT_RAW_REWARDS = False

PICKLE_FILES = {
    "reward": "all_episode_reward.pickle",
    "episode_id": "all_episode_id.pickle",
    "train_step": "all_episode_train_step.pickle",
    "noise": "all_episode_noise.pickle",
    "time": "all_episode_time.pickle",
    "wall_clock": "all_episode_wall_clock.pickle",
    "collision": "all_episode_collision.pickle",
}


def resolve_toplot_dir(raw_input: str, exp_name: str = DEFAULT_EXP_NAME) -> Path:
    """Resolve a run id, run directory, or direct toplot directory."""
    candidate = Path(raw_input).expanduser()

    if candidate.exists():
        candidate = candidate.resolve()
        if candidate.name == "toplot":
            return candidate

        nested_toplot = candidate / "toplot"
        if nested_toplot.is_dir():
            return nested_toplot

        raise FileNotFoundError(
            "Input path exists, but it is neither a toplot directory nor a run "
            "directory containing toplot: {}".format(candidate)
        )

    run_toplot = PROJECT_ROOT / "checkpoints" / exp_name / raw_input / "toplot"
    if run_toplot.is_dir():
        return run_toplot.resolve()

    raise FileNotFoundError(
        "Could not resolve '{}' as a toplot directory or checkpoint run id.".format(raw_input)
    )


def read_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def read_toplot_pickles(toplot_dir: Path) -> Dict[str, List]:
    records = {}
    missing_files = []

    for key, filename in PICKLE_FILES.items():
        path = toplot_dir / filename
        if path.is_file():
            records[key] = read_pickle(path)
        else:
            missing_files.append(filename)

    if "reward" not in records:
        raise FileNotFoundError(
            "Required reward pickle is missing: {}".format(toplot_dir / PICKLE_FILES["reward"])
        )

    if missing_files:
        print("Missing optional pickle files: {}".format(", ".join(missing_files)))

    return records


def read_reward(run_input: str) -> np.ndarray:
    toplot_dir = resolve_toplot_dir(run_input, exp_name=EXP_NAME)
    reward_path = toplot_dir / PICKLE_FILES["reward"]
    if not reward_path.is_file():
        raise FileNotFoundError("Required reward pickle is missing: {}".format(reward_path))
    return np.asarray(read_pickle(reward_path), dtype=float)


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    if window <= 1 or values_array.size < window:
        return values_array
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values_array, kernel, mode="valid")


def print_summary(records: Mapping[str, Sequence]) -> None:
    rewards = np.asarray(records["reward"], dtype=float)
    print("Episodes: {}".format(rewards.size))
    print("Reward mean: {:.6f}".format(float(np.mean(rewards))))
    print("Reward std: {:.6f}".format(float(np.std(rewards))))
    print("Reward min: {:.6f}".format(float(np.min(rewards))))
    print("Reward max: {:.6f}".format(float(np.max(rewards))))
    print("First reward: {:.6f}".format(float(rewards[0])))
    print("Last reward: {:.6f}".format(float(rewards[-1])))

    if "collision" in records:
        collisions = np.asarray(records["collision"], dtype=bool)
        print("Collision rate: {:.2%}".format(float(np.mean(collisions))))

    if "time" in records:
        episode_time = np.asarray(records["time"], dtype=float)
        print("Mean episode steps/time: {:.6f}".format(float(np.mean(episode_time))))


def print_reward_summary(name: str, rewards: Sequence[float]) -> None:
    rewards_array = np.asarray(rewards, dtype=float)
    print("\n{}".format(name))
    print("Episodes: {}".format(rewards_array.size))
    print("Reward mean: {:.6f}".format(float(np.mean(rewards_array))))
    print("Reward std: {:.6f}".format(float(np.std(rewards_array))))
    print("Reward min: {:.6f}".format(float(np.min(rewards_array))))
    print("Reward max: {:.6f}".format(float(np.max(rewards_array))))
    print("First reward: {:.6f}".format(float(rewards_array[0])))
    print("Last reward: {:.6f}".format(float(rewards_array[-1])))


def plot_reward(
    rewards: Sequence[float],
    output_path: Path,
    x_values: Sequence[float] | None = None,
    x_label: str = "Episode",
    moving_average_window: int = 100,
    title: str = "Episode Reward",
    show: bool = False,
) -> None:
    rewards_array = np.asarray(rewards, dtype=float)
    if x_values is None:
        x_array = np.arange(1, rewards_array.size + 1)
    else:
        x_array = np.asarray(x_values, dtype=float)
        if x_array.size != rewards_array.size:
            raise ValueError(
                "x_values length ({}) does not match rewards length ({}).".format(
                    x_array.size,
                    rewards_array.size,
                )
            )

    plt.figure(figsize=(11, 6))
    plt.plot(x_array, rewards_array, color="#4d78a8", linewidth=0.8, alpha=0.45, label="Reward")

    if moving_average_window > 1 and rewards_array.size >= moving_average_window:
        smoothed = moving_average(rewards_array, moving_average_window)
        smoothed_x = x_array[moving_average_window - 1:]
        plt.plot(
            smoothed_x,
            smoothed,
            color="#d45f2a",
            linewidth=2.0,
            label="Moving average ({})".format(moving_average_window),
        )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    if show:
        plt.show(block=True)

    plt.close()


def plot_reward_comparison(
    pcb_biased_att_iddpg_rewards: Sequence[float],
    maddpg_rewards: Sequence[float],
    maac_rewards: Sequence[float],
    iddpg_rewards: Sequence[float],
    mappo_rewards: Sequence[float],
    matd3_rewards: Sequence[float],
    output_path: Path,
    moving_average_window: int = 100,
    show: bool = False,
    plot_raw_rewards: bool = False,
) -> None:
    plt.figure(figsize=(12, 7))

    def plot_series(algo_name: str, rewards: Sequence[float]) -> None:
        rewards_array = np.asarray(rewards, dtype=float)
        x_array = np.arange(1, rewards_array.size + 1)

        if plot_raw_rewards:
            plt.plot(
                x_array,
                rewards_array,
                linewidth=0.6,
                alpha=0.16,
                label="{} raw".format(algo_name),
            )

        if moving_average_window > 1 and rewards_array.size >= moving_average_window:
            y_values = moving_average(rewards_array, moving_average_window)
            x_values = x_array[moving_average_window - 1:]
            line_label = "{} moving average ({})".format(algo_name, moving_average_window)
        else:
            y_values = rewards_array
            x_values = x_array
            line_label = algo_name

        plt.plot(x_values, y_values, linewidth=2.0, label=line_label)

    plot_series("PCB-biased-att-IDDPG", pcb_biased_att_iddpg_rewards)
    plot_series("MADDPG", maddpg_rewards)
    plot_series("MAAC", maac_rewards)
    plot_series("IDDPG", iddpg_rewards)
    plot_series("MAPPO", mappo_rewards)
    plot_series("MATD3", matd3_rewards)

    plt.title("Reward Comparison by Algorithm")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    if show:
        plt.show(block=True)

    plt.close()


def main() -> None:
    if USE_REWARD_COMPARISON:
        pcb_biased_att_iddpg_rewards = read_reward(PCB_BIASED_ATT_IDDPG_RUN)
        maddpg_rewards = read_reward(MADDPG_RUN)
        maac_rewards = read_reward(MAAC_RUN)
        iddpg_rewards = read_reward(IDDPG_RUN)
        mappo_rewards = read_reward(MAPPO_RUN)
        matd3_rewards = read_reward(MATD3_RUN)

        print_reward_summary(
            "PCB-biased-att-IDDPG ({})".format(PCB_BIASED_ATT_IDDPG_RUN),
            pcb_biased_att_iddpg_rewards,
        )
        print_reward_summary("MADDPG ({})".format(MADDPG_RUN), maddpg_rewards)
        print_reward_summary("MAAC ({})".format(MAAC_RUN), maac_rewards)
        print_reward_summary("IDDPG ({})".format(IDDPG_RUN), iddpg_rewards)
        print_reward_summary("MAPPO ({})".format(MAPPO_RUN), mappo_rewards)
        print_reward_summary("MATD3 ({})".format(MATD3_RUN), matd3_rewards)

        output_path = (
            Path(OUTPUT_PATH).expanduser()
            if OUTPUT_PATH
            else PROJECT_ROOT / "plotting_tool" / "reward_comparison.png"
        )
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plot_reward_comparison(
            pcb_biased_att_iddpg_rewards=pcb_biased_att_iddpg_rewards,
            maddpg_rewards=maddpg_rewards,
            maac_rewards=maac_rewards,
            iddpg_rewards=iddpg_rewards,
            mappo_rewards=mappo_rewards,
            matd3_rewards=matd3_rewards,
            output_path=output_path,
            moving_average_window=max(1, int(MOVING_AVERAGE_WINDOW)),
            show=SHOW_PLOT,
            plot_raw_rewards=PLOT_RAW_REWARDS,
        )
        print("\nSaved reward comparison plot to: {}".format(output_path))
        return

    toplot_dir = resolve_toplot_dir(TOPLOT_INPUT, exp_name=EXP_NAME)
    records = read_toplot_pickles(toplot_dir)

    output_path = Path(OUTPUT_PATH).expanduser() if OUTPUT_PATH else toplot_dir / "reward_curve.png"
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Reading pickles from: {}".format(toplot_dir))
    print_summary(records)
    x_values = records.get("train_step")
    x_label = "Training steps" if x_values is not None else "Episode"
    plot_reward(
        records["reward"],
        output_path=output_path,
        x_values=x_values,
        x_label=x_label,
        moving_average_window=max(1, int(MOVING_AVERAGE_WINDOW)),
        title="Episode Reward: {}".format(toplot_dir.parent.name),
        show=SHOW_PLOT,
    )
    print("Saved reward plot to: {}".format(output_path))


if __name__ == "__main__":
    main()
