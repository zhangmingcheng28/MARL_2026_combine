# MARL Project Skeleton - Version 2

Research-oriented structure with:
- shared environment
- train/evaluate split
- command line switch: `--algo`
- experiment name and checkpoint folder
- optional recurrent actor flag: `--use-gru`
- yaml config

## Multiple evaluations

Edit `EVALUATIONS` in `evaluate_multiple.py`. Each dictionary defines one scenario
using the argument names from `arguments.py`, including `repeat_simulation`,
`repeat_od_file`, `n_agents`, `nearest_neighbor_count`, and
`include_building_in_overall_conflict`. Add dictionaries to compare scenarios,
algorithms (`algo`), or checkpoints (`checkpoint_run`, `checkpoint_kind`,
`checkpoint_value`). Unspecified settings inherit `arguments.py` defaults.
Use checkpoints trained with compatible observations; attention actors normally
use `nearest_neighbor_count: 0`.

```powershell
python evaluate_multiple.py --dry-run
python evaluate_multiple.py --repeats 5
python evaluate_multiple.py --config evaluations.json --repeats 2 --continue-on-error
```

`--config` accepts a JSON list with the same dictionaries as `EVALUATIONS`.
`--repeats` defaults to 3; each repeat runs `eval_episodes` episodes (100 by
default). Seeds start at each scenario's `seed` (or the default) and increase by
`--seed-step` (default 1). Use `--seed-step 0` for identical seeds. Replaying the
same OD file with a deterministic policy can produce identical results across
repeats; change the OD file/scenario to compare different routes.

Each batch creates a unique directory under `evaluation_record/batches/`.
Each scenario/repeat preserves the filename
`repeat_OD_for_any_algo_uav_<n_agents>_<algo>_<checkpoint_run>[_nobuilding].pickle`
in the existing evaluation format, plus the resolved `config.json` and
`evaluation.log` (including printed outcome
statistics). The batch `manifest.json` records each attempted run and its status.
Runs execute sequentially in separate processes. A failed run stops the batch
unless `--continue-on-error` is used; any failures result in a nonzero exit code.
`--output-dir` changes the batch parent directory. Dry runs validate arguments
and OD-file existence and print commands without loading models or writing files.

Single evaluations can also use `main.py --evaluation_output_file path/to/result.pickle`.
Use `--no_repeat_simulation` or `--no_include_building_in_overall_conflict` to
explicitly turn off those settings when their defaults are enabled.

## No conflict resolution baseline

Select `no-cr` (alias `no_cr`) in evaluation mode. This deterministic controller
tracks successive planned waypoints at `max_speed`, ignoring other UAVs and
sensor readings when choosing its acceleration. It requires no model checkpoint.
Defaults for the other algorithms are unchanged. In `arguments.py`, select
`"algorithm": "no-cr"` and `"mode": "evaluate"`, or use:

```powershell
python main.py --mode evaluate --algo no-cr --n_agents 8 --eval_episodes 100 --evaluation_output_file evaluation_record/no_cr_uav8.pickle
```

Replay settings still apply. Supply a matching `--repeat_od_file` when changing
the UAV count. Use the same OD file and `astar`/`jps` routes as the other baselines;
CBS introduces coordination during route planning even without reactive avoidance.
The algorithm also works as an `algo` entry in `evaluate_multiple.py`.

The controller ports `get_no_cr_actions` from
`MARL_2nd_paper/80%_sortie_reach_change_map_HS_noCR`: desired velocity points at
the next waypoint, acceleration is `(desired_velocity - velocity) / time_step`,
and its vector magnitude is capped at `acc_max` (8 by default). Actions are
divided by `acc_max` before entering the current environment, which multiplies
them back into physical accelerations. Intermediate waypoints advance at most
once after each simulation step within twice the UAV's protective radius, as
in the legacy circle-overlap test. Final arrival is decided by the shared
environment; the controller never sets arrival or collision flags.

The ordinary evaluator prints collisions by type, arrivals, idle UAVs, and
arrival percentage. It exports the usual trajectory pickle and a neighboring
`<output_stem>_summary.json` containing final counts, success rate, seed, episode
count, and UAV count. Summary JSON files are also written for other algorithms.
No training is required or supported for `no-cr`.

### Comparison with the reference evaluator

| Behavior | Legacy no-CR repository | Current evaluator, including `no-cr` |
| --- | --- | --- |
| Final arrival geometry | UAV circle intersects a waypoint circle of radius `protectiveBound`; nominal center distance is at most twice that radius | UAV circle intersects final-goal circle of radius 1; nominal distance is at most `protectiveBound + 1` |
| Intermediate waypoint | Advance legacy `poping_goal` after a step | Controller maintains its own index with the legacy distance threshold |
| Sortie classification priority | Boundary, building, UAV collision, arrival, idle | Same priority |
| Collided/arrived UAVs, sortie evaluation | Freeze; other UAVs continue | Same behavior |
| Episode termination | Step count exceeds limit, any `done`, or all final arrival flags true | Same, with an added consistency check between arrival flags and `check_goal` |
| Step limit | `100`, tested with strict `< step`, allowing step 101 | Same strict comparison by default |
| Scenarios | Hardcoded 15-UAV OD pickle; 100 evaluation episodes | Configurable OD replay, UAV count, and evaluation episodes |
| Checkpoints | Loads a model during startup even when no-CR bypasses it | `no-cr` skips checkpoint loading |

The flight-control method is ported; identical historical counts are not
guaranteed. Besides final arrival geometry, maps, route planning (legacy JPS
versus current A* default), reward options, and scenario data may differ.
All current algorithms retain the same current final-arrival criterion.
