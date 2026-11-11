"""Run independent evaluations. Edit EVALUATIONS, or supply a JSON list with --config.

Each entry contains a display name and main.py argument names (without '--').
Unspecified options inherit arguments.py defaults. Attention actors normally use
nearest_neighbor_count=0; choose a checkpoint trained with compatible inputs.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

from arguments import build_config, get_args


# Add one dictionary per scenario/algorithm/checkpoint you want to evaluate.
OD_FILES = {
    n_agents: (
        "evaluation_record/batches/20261103_171455_599807/"
        f"{scenario_index:02d}_uav_{n_agents}/run_001/"
        f"repeat_OD_for_any_algo_uav_{n_agents}"
        "_pcb_biased_att_iddpg_280926_09_07_52_nobuilding.pickle"
    )
    for scenario_index, n_agents in enumerate([16, 24, 32], start=1)
}

OD_FILES[50] = (
    "repeat_OD_for_any_algo_uav_50"
    "_pcb_biased_att_iddpg_131026_03_26_09_nobuilding.pickle"
)

# Paths taken from reward_plotting.py.
MODELS = [
    # {
    #     "algo": "maac",
    #     "checkpoint_dir": r"D:\server_v100_simulation_data",
    #     "exp_name": "simulation_data",
    #     "checkpoint_run": "081026_06_22_24",
    # },
    # {
    #     "algo": "matd3",
    #     "checkpoint_dir": r"D:\server_v100_simulation_data",
    #     "exp_name": "simulation_data",
    #     "checkpoint_run": "081026_06_26_05",
    # },
    # {
    #     "algo": "maddpg",
    #     "checkpoint_dir": "checkpoints",
    #     "exp_name": "default_exp",
    #     "checkpoint_run": "201026_01_59_59",
    # },
    {"algo": "no-cr"},
]

EVALUATIONS = [
    {
        **model,
        "name": f"{model['algo']}_uav_{n_agents}",
        # "checkpoint_kind": "step",
        # "checkpoint_value": 450000,
        "repeat_simulation": True,
        "repeat_od_file": OD_FILES[n_agents],
        "n_agents": n_agents,
        # "nearest_neighbor_count": (
        #     7 if model["algo"] in ("maac", "matd3") else 3
        # ),
        "include_building_in_overall_conflict": False,
        "eval_episodes": 100,
        "seed": 777,
    }
    for model in MODELS
    for n_agents in [16, 24, 32, 50]
]

PROJECT_ROOT = Path(__file__).resolve().parent
NEGATIVE_FLAGS = {"repeat_simulation", "include_building_in_overall_conflict"}


def scenario_arguments(scenario):
    """Translate the editable settings to the same CLI used by main.py."""
    result = ["--mode", "evaluate"]
    for key, value in scenario.items():
        if key == "name":
            continue
        if key in {"mode", "evaluation_output_file"}:
            raise ValueError("{} is controlled by the batch runner.".format(key))
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ValueError("Invalid argument name: {!r}".format(key))
        if isinstance(value, bool):
            if value:
                result.append("--" + key)
            elif key in NEGATIVE_FLAGS:
                result.append("--no_" + key)
            else:
                raise ValueError("Explicit False is only supported for {}.".format(sorted(NEGATIVE_FLAGS)))
        elif value is None:
            raise ValueError("Omit {} to use its default instead of null.".format(key))
        else:
            if isinstance(value, list):
                value = ",".join(map(str, value))
            result.extend(["--" + key, str(value)])
    return result


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON list of scenario dictionaries, replacing EVALUATIONS")
    parser.add_argument("--repeats", type=positive_int, default=1, help="full evaluations per scenario (default: 3)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "evaluation_record" / "batches")
    parser.add_argument("--seed-step", type=int, default=1, help="seed increment per repeat; 0 reuses the seed")
    parser.add_argument("--dry-run", action="store_true", help="validate settings and print commands without running")
    parser.add_argument("--continue-on-error", action="store_true", help="finish remaining runs if a run fails")
    options = parser.parse_args(argv)

    scenarios = EVALUATIONS
    if options.config:
        with options.config.open(encoding="utf-8") as handle:
            scenarios = json.load(handle)
    if not isinstance(scenarios, list) or not scenarios:
        parser.error("Evaluations must be a nonempty list of dictionaries.")

    # Validate every scenario before starting any costly evaluations.
    prepared = []
    for index, scenario in enumerate(scenarios, 1):
        if not isinstance(scenario, dict):
            parser.error("Scenario {} must be a dictionary.".format(index))
        try:
            cli = scenario_arguments(scenario)
        except ValueError as error:
            parser.error(str(error))
        args = get_args(cli)
        if args.n_agents < 1 or args.eval_episodes < 1 or args.nearest_neighbor_count < 0:
            parser.error("n_agents/eval_episodes must be positive and nearest_neighbor_count nonnegative.")
        config = build_config(args)
        if args.repeat_simulation and not Path(config["paths"]["repeat_od_file"] or "").is_file():
            parser.error("Repeat OD file does not exist: {}".format(config["paths"]["repeat_od_file"]))
        name = str(scenario.get("name", "scenario_{}".format(index)))
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "scenario"
        pickle_name = "repeat_OD_for_any_algo_uav_{}_{}_{}".format(
            config["env"]["n_agents"], config["algorithm"], config["paths"]["checkpoint_run"]
        )
        if not config["flags"]["include_building_in_overall_conflict"]:
            pickle_name = "{}_nobuilding".format(pickle_name)
        pickle_name = "{}.pickle".format(pickle_name)
        prepared.append((index, slug, cli, args.seed, pickle_name))

    batch_dir = options.output_dir.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    records = []
    for index, slug, cli, base_seed, pickle_name in prepared:
        for repeat in range(options.repeats):
            run_dir = batch_dir / "{:02d}_{}".format(index, slug) / "run_{:03d}".format(repeat + 1)
            output_file = run_dir / pickle_name
            seed = base_seed + repeat * options.seed_step
            run_cli = cli + ["--seed", str(seed), "--evaluation_output_file", str(output_file)]
            resolved_config = build_config(get_args(run_cli))
            command = [sys.executable, "-u", str(PROJECT_ROOT / "main.py")] + run_cli
            print("[BATCH] {} repeat {}/{} (seed {})".format(slug, repeat + 1, options.repeats, seed), flush=True)
            if options.dry_run:
                print(subprocess.list2cmdline(command))
                continue

            run_dir.mkdir(parents=True, exist_ok=False)
            with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
                json.dump(resolved_config, handle, indent=2)
            log_file = run_dir / "evaluation.log"
            print("[BATCH] Log: {}".format(log_file), flush=True)
            with log_file.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
            succeeded = completed.returncode == 0 and output_file.is_file()
            records.append({
                "scenario": slug, "repeat": repeat + 1, "seed": seed,
                "returncode": completed.returncode, "status": "completed" if succeeded else "failed",
                "output_file": str(output_file), "log_file": str(log_file),
            })
            with (batch_dir / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(records, handle, indent=2)
            print("[BATCH] {}: {}".format(records[-1]["status"], run_dir), flush=True)
            if not succeeded and not options.continue_on_error:
                return 1

    if options.dry_run:
        print("[BATCH] Dry run complete; no files written.")
    else:
        print("[BATCH] Results: {}".format(batch_dir))
    return int(any(record["status"] == "failed" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
