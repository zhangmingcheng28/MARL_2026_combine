# -*- coding: utf-8 -*-
r"""
Read training pickles from a run's ``toplot`` directory and plot rewards.

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
EXP_NAME = DEFAULT_EXP_NAME
MOVING_AVERAGE_WINDOW = 500
SHOW_PLOT = False
OUTPUT_PATH = None  # None saves to <toplot>/reward_curve.png

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
        plt.show()

    plt.close()


def main() -> None:
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
