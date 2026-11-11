# -*- coding: utf-8 -*-
r"""
Plot training rewards and recorded evaluation success rates for RUN_INPUTS.

Run this file directly. Set USE_REWARD_COMPARISON=False for TOPLOT_INPUT.
Source directories are read only; plots and evaluation CSVs go to plotting_tool.
"""
from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP_NAME = "default_exp"

# Quick-look input. This can be either:
#   1. a run id, for example "020626_05_47_51"
#   2. a full run directory
#   3. a full toplot directory
TOPLOT_INPUT = r"F:\githubClone\MARL_2026_combine\checkpoints\default_exp\280926_09_07_52"

# Reward comparison inputs. Set USE_REWARD_COMPARISON = False to use TOPLOT_INPUT
# for a single-run plot.
USE_REWARD_COMPARISON = True
PCB_BIASED_ATT_IDDPG_RUN = r"F:\githubClone\MARL_2026_combine\checkpoints\default_exp\280926_09_07_52"
# PCB_BIASED_ATT_IDDPG_RUN = r"F:\githubClone\MARL_2026_combine\checkpoints\default_exp\131026_03_26_09" (DO NOT USE THIS is model is NOT good)
MAAC_RUN = r"D:\server_v100_simulation_data\simulation_data\081026_06_22_24"
MATD3_RUN = r"D:\server_v100_simulation_data\simulation_data\081026_06_26_05"
MADDPG_RUN = r"F:\githubClone\MARL_2026_combine\checkpoints\default_exp\201026_01_59_59"
RUN_INPUTS = {
    "PCB-biased-att-IDDPG": PCB_BIASED_ATT_IDDPG_RUN,
    "MAAC": MAAC_RUN,
    "MATD3": MATD3_RUN,
    "MADDPG": MADDPG_RUN,
}

EXP_NAME = DEFAULT_EXP_NAME
# Edit these manually, then run this file again. Set a window to 1 for no smoothing.
MOVING_AVERAGE_WINDOW = 100  # Reward window: number of completed episodes.
SUCCESS_MOVING_AVERAGE_WINDOW = 5  # Success window: number of evaluation checkpoints.
PLOT_VARIABILITY_BAND = True  # Local mean +/- standard deviation within each run.
VARIABILITY_STD_SCALE = 1.0
VARIABILITY_BAND_ALPHA = 0.12
COMBINED_LAYOUT = "stacked"  # Separate reward/success panels; "overlay" uses two y-axes.
# Zoom window inside the success-rate axes. Position is (left, bottom, width, height)
# in fractions of the parent axes. These settings apply to standalone and combined plots.
SHOW_SUCCESS_INSET = True
SUCCESS_INSET_XLIM = (440000, 450000)  # Final training-step interval to enlarge.
SUCCESS_INSET_YLIM = (82, 88)  # Focus on the leading means; None fits all zoomed data.
SUCCESS_INSET_BOUNDS = (0.67, 0.08, 0.27, 0.48)
SUCCESS_INSET_COLOR = "#c000a0"
SUCCESS_INSET_SHOW_BAND = False  # Keep the main shading; clear the close-up for readability.
SUCCESS_INSET_LABEL_ENDPOINTS = True  # Label plotted rolling means, not raw evaluation results.
SHOW_PLOT = True
OUTPUT_PATH = None  # Default: plotting_tool/reward_comparison.png (reward_curve.png for one run)
SUCCESS_OUTPUT_PATH = None  # Default: plotting_tool/success_rate_comparison.png
SUCCESS_CSV_PATH = None  # Default: plotting_tool/success_rate_comparison.csv
COMBINED_OUTPUT_PATH = None  # Default: plotting_tool/reward_and_success_comparison.png
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
    rewards = np.asarray(read_pickle(reward_path), dtype=float)
    if rewards.ndim != 1 or rewards.size == 0 or not np.all(np.isfinite(rewards)):
        raise ValueError("Expected a nonempty, finite reward series: {}".format(reward_path))
    return rewards


def _evaluation_point(record: Mapping, source: Path, final_only: bool = False) -> dict:
    """Normalize the recorded outcome counts, without assuming an episode count."""
    def count(key: str, legacy_key: str) -> int:
        value = record[key] if key in record else record[legacy_key]
        number = float(value)
        if not np.isfinite(number) or number < 0 or not number.is_integer():
            raise ValueError("Invalid {} in {}: {}".format(key, source, value))
        return int(number)

    reached = count("destination_reached", "Destination reached")
    collisions = count("total_collision", "Total collision")
    idle = count("idle_uav", "Idle UAV")
    total = reached + collisions + idle
    step = float(record["train_step"])
    if total == 0 or not np.isfinite(step) or step < 0 or not step.is_integer():
        raise ValueError("Invalid evaluation total or training step in {}".format(source))
    return {
        "train_step": int(step),
        "destination_reached": reached,
        "total_sorties": total,
        "success_rate_percent": 100.0 * reached / total,
        "record_kind": "final_only" if final_only else "intermediate_evaluation",
        "source": str(source),
    }


def read_success_history(run_input: str) -> List[dict]:
    """Read evaluation history; a final summary is a single point, never a curve."""
    toplot_dir = resolve_toplot_dir(run_input, exp_name=EXP_NAME)
    history_path = toplot_dir / "train_eval_record.pickle"
    if history_path.is_file():
        records = read_pickle(history_path)
        if not isinstance(records, (list, tuple)) or not records:
            raise ValueError("Expected nonempty evaluation records: {}".format(history_path))
        points = [_evaluation_point(record, history_path) for record in records]
        return sorted(points, key=lambda point: point["train_step"])

    summary_path = toplot_dir.parent / "training_eval_summary.txt"
    if not summary_path.is_file():
        print("No evaluation history or final summary for {}".format(toplot_dir.parent))
        return []
    summary = summary_path.read_text(encoding="utf-8-sig")
    final_record = {}
    for key, label in (
        ("train_step", "total_steps_used:"),
        ("destination_reached", "Destination reached"),
        ("total_collision", "Total collision"),
        ("idle_uav", "Idle UAV"),
    ):
        matches = re.findall(r"^\s*" + re.escape(label) + r"\s*(\d+)\s*$", summary, re.MULTILINE)
        if not matches:
            print("No complete evaluation result in {} (missing {}).".format(summary_path, label))
            return []
        final_record[key] = int(matches[-1])
    print("No intermediate evaluation history for {}; plotting its final result only.".format(toplot_dir.parent))
    return [_evaluation_point(final_record, summary_path, final_only=True)]


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    if window <= 1 or values_array.size < window:
        return values_array
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values_array, kernel, mode="valid")


def rolling_mean_and_std(values: Sequence[float], window: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Full trailing-window statistics; bands represent local variability, not seed uncertainty."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Rolling statistics require a nonempty, finite 1D series.")
    window = min(max(1, int(window)), values.size)
    if window == 1:
        return values.copy(), np.zeros_like(values), window
    # Shift the series to limit cancellation when computing variance from moments.
    shifted = values - values[0]
    cumulative = np.concatenate(([0.0], np.cumsum(shifted)))
    cumulative_squared = np.concatenate(([0.0], np.cumsum(shifted * shifted)))
    means_shifted = (cumulative[window:] - cumulative[:-window]) / window
    second_moments = (cumulative_squared[window:] - cumulative_squared[:-window]) / window
    deviations = np.sqrt(np.maximum(second_moments - means_shifted * means_shifted, 0.0))
    return means_shifted + values[0], deviations, window


def _plot_mean_band(
    axis: plt.Axes, x_values: Sequence[float], values: Sequence[float], window: int,
    color: str, label: str, raw: bool = False, linestyle: str = "-", percentage: bool = False,
    show_band: bool | None = None,
) -> None:
    values = np.asarray(values, dtype=float)
    x_values = np.asarray(x_values, dtype=float)
    if x_values.shape != values.shape or not np.all(np.isfinite(x_values)) or np.any(np.diff(x_values) < 0):
        raise ValueError("Invalid coordinates for {}".format(label))
    means, deviations, effective_window = rolling_mean_and_std(values, window)
    mean_x = x_values[effective_window - 1:]
    if raw:
        axis.plot(x_values, values, color=color, linewidth=0.5, alpha=0.12, label="_nolegend_")
    band_enabled = PLOT_VARIABILITY_BAND if show_band is None else show_band
    if band_enabled:
        scale = float(VARIABILITY_STD_SCALE)
        alpha = float(VARIABILITY_BAND_ALPHA)
        if not np.isfinite(scale) or scale < 0 or not 0 <= alpha <= 1:
            raise ValueError("Band scale must be finite and nonnegative; alpha must be between 0 and 1.")
        lower = means - scale * deviations
        upper = means + scale * deviations
        if percentage:
            lower, upper = np.clip(lower, 0, 100), np.clip(upper, 0, 100)
        axis.fill_between(mean_x, lower, upper, color=color, alpha=alpha, linewidth=0,
                          label="_nolegend_", zorder=1)
    axis.plot(mean_x, means, color=color, linewidth=2.0, linestyle=linestyle, label=label, zorder=3)


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
) -> plt.Figure:
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

    figure, axis = plt.subplots(figsize=(11, 6))
    _plot_mean_band(axis, x_array, rewards_array, moving_average_window, "C0", "Reward mean",
                    raw=PLOT_RAW_REWARDS)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    if show:
        plt.show(block=True)

        plt.close(figure)
    return figure


def plot_reward_comparison(
    rewards_by_algorithm: Mapping[str, Sequence[float]],
    output_path: Path | None,
    moving_average_window: int = 100,
    show: bool = False,
    plot_raw_rewards: bool = False,
    x_values_by_algorithm: Mapping[str, Sequence[float]] | None = None,
    axis: plt.Axes | None = None,
) -> plt.Figure:
    if axis is None:
        figure, axis = plt.subplots(figsize=(12, 7))
    else:
        figure = axis.figure

    def plot_series(algo_name: str, rewards: Sequence[float], color: str) -> None:
        rewards_array = np.asarray(rewards, dtype=float)
        x_array = (
            np.asarray(x_values_by_algorithm[algo_name], dtype=float)
            if x_values_by_algorithm is not None
            else np.arange(1, rewards_array.size + 1)
        )
        if x_array.shape != rewards_array.shape or not np.all(np.isfinite(x_array)) or np.any(np.diff(x_array) < 0):
            raise ValueError("Invalid reward coordinates for {}".format(algo_name))

        _plot_mean_band(axis, x_array, rewards_array, moving_average_window, color, algo_name,
                        raw=plot_raw_rewards)

    for index, (name, rewards) in enumerate(rewards_by_algorithm.items()):
        plot_series(name, rewards, "C{}".format(index % 10))

    axis.set(title="Reward Mean (window: {} episodes)".format(moving_average_window), ylabel="Reward",
             xlabel="Training steps" if x_values_by_algorithm is not None else "Completed training episodes")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    axis.legend(fontsize=9)
    if output_path is not None:
        figure.tight_layout()
        figure.savefig(output_path, dpi=300)

    if show:
        plt.show(block=True)

        plt.close(figure)
    return figure


def _plot_success_series(
    axis: plt.Axes, histories: Mapping[str, Sequence[Mapping]], linestyle: str,
    show_band: bool | None = None,
) -> List[str]:
    """Use exactly the same mean, bands, and colors in the main plot and zoom."""
    missing = []
    for index, (name, points) in enumerate(histories.items()):
        color = "C{}".format(index % 10)
        if not points:
            missing.append(name)
            continue
        steps = [point["train_step"] for point in points]
        rates = [point["success_rate_percent"] for point in points]
        if points[0]["record_kind"] == "final_only":
            axis.scatter(steps, rates, color=color, marker="D", s=80,
                         label="{} (final only; no intermediate records)".format(name), zorder=4)
        else:
            _plot_mean_band(axis, steps, rates, SUCCESS_MOVING_AVERAGE_WINDOW, color, name,
                            linestyle=linestyle, percentage=True, show_band=show_band)
    return missing


def _add_success_inset(
    axis: plt.Axes, histories: Mapping[str, Sequence[Mapping]], linestyle: str,
) -> None:
    x_limits = np.asarray(SUCCESS_INSET_XLIM, dtype=float)
    bounds = np.asarray(SUCCESS_INSET_BOUNDS, dtype=float)
    if x_limits.shape != (2,) or not np.all(np.isfinite(x_limits)) or x_limits[0] >= x_limits[1]:
        raise ValueError("SUCCESS_INSET_XLIM must contain two increasing, finite training steps.")
    if (bounds.shape != (4,) or not np.all(np.isfinite(bounds)) or np.any(bounds < 0)
            or np.any(bounds[2:] <= 0) or np.any(bounds[:2] + bounds[2:] > 1)):
        raise ValueError("SUCCESS_INSET_BOUNDS must define a positive rectangle inside the axes.")

    # Test the actual plotted means, so the inset never shows unrelated data or
    # recomputes smoothing after slicing the selected interval.
    visible_values = []
    for line in axis.lines:
        xs, ys = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        visible_values.extend(ys[(xs >= x_limits[0]) & (xs <= x_limits[1])])
    for points in histories.values():
        visible_values.extend(point["success_rate_percent"] for point in points
                              if point["record_kind"] == "final_only"
                              and x_limits[0] <= point["train_step"] <= x_limits[1])
    if not visible_values:
        print("No plotted success records inside {}; skipping zoom inset.".format(tuple(x_limits)))
        return

    if SUCCESS_INSET_YLIM is None:
        lower, upper = min(visible_values), max(visible_values)
        padding = max(2.5, 0.1 * (upper - lower))
        y_limits = (max(0, lower - padding), min(100, upper + padding))
    else:
        y_limits = np.asarray(SUCCESS_INSET_YLIM, dtype=float)
        if (y_limits.shape != (2,) or not np.all(np.isfinite(y_limits))
                or not 0 <= y_limits[0] < y_limits[1] <= 100):
            raise ValueError("SUCCESS_INSET_YLIM must be None or increasing percentages between 0 and 100.")

    inset = axis.inset_axes(bounds, zorder=6)
    inset.set_in_layout(False)
    _plot_success_series(inset, histories, linestyle,
                         show_band=PLOT_VARIABILITY_BAND and SUCCESS_INSET_SHOW_BAND)
    inset.set(xlim=x_limits, ylim=y_limits, title="Final rolling means")
    inset.title.set_fontsize(9)
    inset.set_xlabel("Training steps", fontsize=7, labelpad=1)
    inset.set_ylabel("Success (%)", fontsize=7, labelpad=1)
    inset.tick_params(labelsize=7)
    inset.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useOffset=False)
    inset.xaxis.get_offset_text().set_fontsize(7)
    inset.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    if SUCCESS_INSET_LABEL_ENDPOINTS:
        outside_notes = []
        for line in inset.lines:
            xs, ys = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
            if not xs.size or not x_limits[0] <= xs[-1] <= x_limits[1]:
                continue
            name = line.get_label().replace("PCB-biased-att-IDDPG", "PCB")
            rate = float(ys[-1])
            label = "{}: {:.2f}%".format(name, rate)
            if y_limits[0] <= rate <= y_limits[1]:
                inset.scatter([xs[-1]], [rate], s=22, color=line.get_color(),
                              edgecolors="white", linewidths=0.5, clip_on=False, zorder=7)
                inset.annotate(label, xy=(xs[-1], rate), xytext=(-5, 0),
                               textcoords="offset points", ha="right", va="center",
                               color=line.get_color(), fontsize=8, fontweight="bold", zorder=8,
                               bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.5})
            else:
                direction = "Below" if rate < y_limits[0] else "Above"
                outside_notes.append("{} zoom: {}".format(direction, label))
        if outside_notes:
            inset.text(0.03, 0.03, "\n".join(outside_notes), transform=inset.transAxes,
                       fontsize=7, va="bottom", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    for spine in inset.spines.values():
        spine.set_edgecolor(SUCCESS_INSET_COLOR)
        spine.set_linewidth(1.2)
    selection = Rectangle(
        (x_limits[0], y_limits[0]), x_limits[1] - x_limits[0], y_limits[1] - y_limits[0],
        fill=False, edgecolor=SUCCESS_INSET_COLOR, linewidth=1.2, zorder=5,
    )
    selection.set_in_layout(False)
    axis.add_patch(selection)
    # A short arrow from the highlighted region to the enlargement is less
    # intrusive than diagonal connectors spanning the learning curves.
    arrow = axis.annotate(
        "", xy=(bounds[0] + 0.9 * bounds[2], bounds[1] + bounds[3] + 0.015),
        xycoords="axes fraction", xytext=(x_limits[1], y_limits[0]), textcoords="data",
        arrowprops={"arrowstyle": "->", "color": SUCCESS_INSET_COLOR, "lw": 1.2}, zorder=5,
    )
    arrow.set_in_layout(False)


def plot_success_comparison(
    histories: Mapping[str, Sequence[Mapping]], output_path: Path | None, show: bool = False,
    axis: plt.Axes | None = None,
    linestyle: str = "-",
) -> plt.Figure:
    if axis is None:
        figure, axis = plt.subplots(figsize=(12, 7))
    else:
        figure = axis.figure
    missing = _plot_success_series(axis, histories, linestyle)
    if missing:
        axis.text(0.02, 0.02, "No evaluation records: " + ", ".join(missing),
                  transform=axis.transAxes, fontsize=9)
    axis.set(title="Success Rate Mean (window: {} checkpoints)".format(SUCCESS_MOVING_AVERAGE_WINDOW), xlabel="Training steps",
             ylabel="Destination-reaching success rate (%)", ylim=(0, 100))
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    if any(histories.values()):
        axis.legend(loc="upper left")
        if SHOW_SUCCESS_INSET:
            _add_success_inset(axis, histories, linestyle)
    if output_path is not None:
        figure.tight_layout()
        figure.savefig(output_path, dpi=300)
    if show:
        plt.show(block=True)
        plt.close(figure)
    return figure


def plot_reward_and_success(
    rewards_by_algorithm: Mapping[str, Sequence[float]],
    histories: Mapping[str, Sequence[Mapping]],
    output_path: Path,
    x_values_by_algorithm: Mapping[str, Sequence[float]] | None = None,
) -> plt.Figure:
    if COMBINED_LAYOUT not in ("overlay", "stacked"):
        raise ValueError("COMBINED_LAYOUT must be 'overlay' or 'stacked'.")
    overlay = COMBINED_LAYOUT == "overlay" and x_values_by_algorithm is not None
    if overlay:
        figure, reward_axis = plt.subplots(figsize=(14, 8))
        axes = [reward_axis, reward_axis.twinx()]
    else:
        # Legacy rewards without recorded steps need separate x-axes.
        figure, axes = plt.subplots(2, 1, figsize=(13, 10),
                                   sharex=x_values_by_algorithm is not None)
    plot_reward_comparison(
        rewards_by_algorithm, output_path=None, axis=axes[0],
        x_values_by_algorithm=x_values_by_algorithm,
        moving_average_window=max(1, int(MOVING_AVERAGE_WINDOW)),
        plot_raw_rewards=PLOT_RAW_REWARDS,
    )
    plot_success_comparison(histories, output_path=None, axis=axes[1],
                            linestyle="--" if overlay else "-")
    if overlay:
        axes[0].set_title("")
        axes[1].set_title("")
        axes[1].grid(False)
        axes[0].legend(fontsize=9, loc="upper left", title="Reward mean (solid)")
        axes[1].legend(fontsize=9, loc="lower right", title="Success rate mean (dashed)")
    else:
        if x_values_by_algorithm is not None:
            axes[0].set_xlabel("")
        axes[0].legend(fontsize=9, ncol=2)
        axes[1].legend(fontsize=9, loc="upper left")
        axes[0].set_title("Training Reward")
        axes[1].set_title("Evaluation Success Rate")
    figure.suptitle("Training Reward and Evaluation Success", fontsize=14)
    band_note = (
        "Shading: +/- {} local standard deviation (within each run).".format(VARIABILITY_STD_SCALE)
        if PLOT_VARIABILITY_BAND else "Shading disabled."
    )
    figure.text(0.5, 0.015, "Mean windows: {} reward episodes; {} evaluation checkpoints. {}".format(
        MOVING_AVERAGE_WINDOW, SUCCESS_MOVING_AVERAGE_WINDOW, band_note), ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 0.97))
    figure.savefig(output_path, dpi=300)
    return figure


def write_success_csv(histories: Mapping[str, Sequence[Mapping]], output_path: Path) -> None:
    fields = ["algorithm", "train_step", "destination_reached", "total_sorties",
              "success_rate_percent", "record_kind", "source"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, points in histories.items():
            for point in points:
                writer.writerow({"algorithm": name, **point})


def _output_path(configured_path: str | None, filename: str) -> Path:
    path = Path(configured_path).expanduser() if configured_path else PROJECT_ROOT / "plotting_tool" / filename
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    if USE_REWARD_COMPARISON:
        runs = RUN_INPUTS
    else:
        toplot_dir = resolve_toplot_dir(TOPLOT_INPUT, exp_name=EXP_NAME)
        runs = {toplot_dir.parent.name: TOPLOT_INPUT}

    rewards_by_algorithm = {}
    reward_steps = {}
    histories = {}
    for name, run_input in runs.items():
        print("\nReading {} from: {}".format(name, resolve_toplot_dir(run_input, EXP_NAME)))
        rewards_by_algorithm[name] = read_reward(run_input)
        step_path = resolve_toplot_dir(run_input, EXP_NAME) / PICKLE_FILES["train_step"]
        if step_path.is_file():
            reward_steps[name] = np.asarray(read_pickle(step_path), dtype=float)
        print_reward_summary(name, rewards_by_algorithm[name])
        histories[name] = read_success_history(run_input)
        if histories[name]:
            last = histories[name][-1]
            print("Evaluation points: {}; latest: {}/{} = {:.3f}% at step {} ({})".format(
                len(histories[name]), last["destination_reached"], last["total_sorties"],
                last["success_rate_percent"], last["train_step"], last["record_kind"]))

    reward_path = _output_path(OUTPUT_PATH, "reward_comparison.png" if USE_REWARD_COMPARISON else "reward_curve.png")
    success_path = _output_path(SUCCESS_OUTPUT_PATH, "success_rate_comparison.png" if USE_REWARD_COMPARISON else "success_rate_curve.png")
    csv_path = _output_path(SUCCESS_CSV_PATH, "success_rate_comparison.csv" if USE_REWARD_COMPARISON else "success_rate_curve.csv")
    combined_path = _output_path(COMBINED_OUTPUT_PATH, "reward_and_success_comparison.png" if USE_REWARD_COMPARISON else "reward_and_success_curve.png")
    reward_coordinates = reward_steps if len(reward_steps) == len(runs) else None
    if reward_coordinates is None:
        print("Some reward records lack training steps; the reward panel uses completed episodes with an independent x-axis.")

    if USE_REWARD_COMPARISON:
        reward_figure = plot_reward_comparison(
            rewards_by_algorithm, output_path=reward_path,
            moving_average_window=max(1, int(MOVING_AVERAGE_WINDOW)),
            plot_raw_rewards=PLOT_RAW_REWARDS,
            x_values_by_algorithm=reward_coordinates,
        )
    else:
        records = read_toplot_pickles(toplot_dir)
        x_values = records.get("train_step")
        reward_figure = plot_reward(
            records["reward"], output_path=reward_path, x_values=x_values,
            x_label="Training steps" if x_values is not None else "Completed training episodes",
            moving_average_window=max(1, int(MOVING_AVERAGE_WINDOW)),
            title="Episode Reward: {}".format(toplot_dir.parent.name),
        )
    success_figure = plot_success_comparison(histories, success_path)
    combined_figure = plot_reward_and_success(
        rewards_by_algorithm, histories, combined_path, reward_coordinates,
    )
    write_success_csv(histories, csv_path)
    print("\nSaved reward plot to: {}".format(reward_path))
    print("Saved success-rate plot to: {}".format(success_path))
    print("Saved evaluation data to: {}".format(csv_path))
    print("Saved combined plot to: {}".format(combined_path))
    # Keep the standalone exports, and display the requested combined figure.
    plt.close(reward_figure)
    plt.close(success_figure)
    try:
        if SHOW_PLOT:
            plt.show(block=True)
    finally:
        plt.close(combined_figure)


if __name__ == "__main__":
    main()
