import csv
import os
import pickle
import time
from copy import deepcopy
from types import SimpleNamespace
import numpy as np

from envs.shared_env import SharedMultiAgentEnv
from envs.vector_env import SubprocVecEnv
from agents import build_trainer
from config.paths import create_training_run_dirs
from utils.plotting_helper import *


def _build_plot_env_from_snapshot(template_env, agent_snapshot):
    if agent_snapshot:
        map_idx = agent_snapshot[0].get("map_idx")
        if map_idx is not None and hasattr(template_env, "_activate_precomputed_map"):
            template_env._activate_precomputed_map(map_idx)
    plot_agents = {}
    for agent_info in agent_snapshot:
        goal_points = list(agent_info["goal"])
        plot_agents[agent_info["agent_idx"]] = SimpleNamespace(
            agent_name=agent_info["agent_name"],
            ini_pos=np.asarray(agent_info["ini_pos"]),
            goal=goal_points,
            heading=agent_info["heading"],
            protectiveBound=agent_info["protectiveBound"],
            detectionRange=agent_info["detectionRange"],
            reach_target=agent_info["reach_target"],
            ref_line=SimpleNamespace(coords=[tuple(agent_info["ini_pos"])] + goal_points),
        )
    template_env.all_uavs = plot_agents
    return template_env


def _start_episode_run(env_slot, trainer, episode_id, backend):
    episode_start_time = time.perf_counter()
    if hasattr(trainer, "begin_episode"):
        trainer.begin_episode(episode_id)
    if backend == "subproc":
        cur_state, norm_cur_state, agent_snapshot = env_slot["vec_env"].reset_at(env_slot["env_idx"], episode_id)
        n_agents = env_slot["n_agents"]
    else:
        env = env_slot["env"]
        cur_state, norm_cur_state = env.reset(episode_id, show=0)
        agent_snapshot = None
        n_agents = env.n_agents
    return {
        "env_slot": env_slot,
        "episode_id": episode_id,
        "episode_start_time": episode_start_time,
        "cur_state": cur_state,
        "norm_cur_state": norm_cur_state,
        "episode_reward": 0.0,
        "episode_decision": [False] * 3,
        "episode_goal_found": [False] * n_agents,
        "trajectory_eachPlay": [],
        "single_eps_critic_cal_record": [],
        "agent_snapshot": agent_snapshot,
        "initial_agent_snapshot": agent_snapshot,
    }


def _finalize_episode_run(run, env_cfg, info, dones, current_step):
    episode_decision = run["episode_decision"]
    episode_goal_found = run["episode_goal_found"]

    if env_cfg["max_steps"] < current_step:
        episode_decision[0] = True
        print(
            "Agents stuck in some places, maximum step in one episode reached, current episode {} ends, all {} steps used".format(
                run["episode_id"], env_cfg["max_steps"]
            )
        )
    elif True in dones:
        episode_decision[1] = True
        print(
            "Some agent triggers termination condition like collision, current episode {} ends at step {}".format(
                run["episode_id"], current_step - 1
            )
        )
    elif all(info["check_goal"]):
        episode_decision[2] = True
        print(
            "All agents have reached their destinations at step {}, episode {} terminated.".format(
                current_step - 1, run["episode_id"]
            )
        )

    if True in episode_decision:
        if run["agent_snapshot"] is not None:
            for agent_info in run["agent_snapshot"]:
                episode_goal_found[agent_info["agent_idx"]] = agent_info["reach_target"]
        else:
            env = run["env_slot"]["env"]
            for agent_idx, agent in env.all_uavs.items():
                episode_goal_found[agent_idx] = agent.reach_target
        return True

    return False


def _wandb_scalar(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return None
    return None


def _wandb_log(wandb_module, data, step):
    if wandb_module is None:
        return
    payload = {}
    for key, value in data.items():
        scalar = _wandb_scalar(value)
        if scalar is not None:
            payload[key] = scalar
    if payload:
        wandb_module.log(payload, step=step)


def _prefixed_metrics(prefix, source):
    if not source:
        return {}
    metrics = {}
    for key, value in source.items():
        scalar = _wandb_scalar(value)
        if scalar is not None:
            metrics[f"{prefix}/{key}"] = scalar
    return metrics


def _safe_float_array(values):
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return np.asarray([], dtype=np.float64)


def _numeric_value(value):
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.size != 1:
        return None
    return float(arr.reshape(-1)[0])


def _status_holder_wandb_metrics(info):
    status_holder = info.get("status_holder", [])
    if not status_holder:
        return {}

    keys = [
        "Euclidean_dist_to_goal",
        "goal_leading_reward",
        "deviation_to_ref_line",
        "deviation_to_ref_line_reward",
        "near_building_penalty",
        "small_step_penalty",
        "current_drone_speed",
        "addition_near_goal_reward",
        "segment_reward",
        "near_drone_penalty",
    ]
    metrics = {}
    for key in keys:
        values = []
        for agent_idx, agent_status in enumerate(status_holder):
            if key not in agent_status:
                continue
            value = _numeric_value(agent_status[key])
            if value is None:
                continue
            values.append(value)
            metrics[f"agent_{agent_idx}/{key}"] = value
        if values:
            values_arr = np.asarray(values, dtype=np.float64)
            metrics[f"env_status/{key}_mean"] = float(np.mean(values_arr))
            metrics[f"env_status/{key}_min"] = float(np.min(values_arr))
            metrics[f"env_status/{key}_max"] = float(np.max(values_arr))
    return metrics


def _agent_step_wandb_metrics(actions, rewards_arr, dones_arr, check_goal):
    metrics = {}
    flat_rewards = rewards_arr.reshape(-1) if rewards_arr.size else []
    flat_dones = dones_arr.reshape(-1) if dones_arr.size else []
    action_rows = actions if actions.ndim == 2 else np.asarray([], dtype=np.float64)

    for agent_idx, reward in enumerate(flat_rewards):
        metrics[f"agent_{agent_idx}/reward"] = float(reward)
    for agent_idx, done in enumerate(flat_dones):
        metrics[f"agent_{agent_idx}/done"] = int(bool(done))
    for agent_idx, reached in enumerate(check_goal):
        metrics[f"agent_{agent_idx}/reached"] = int(bool(reached))
    for agent_idx, action in enumerate(action_rows):
        metrics[f"agent_{agent_idx}/action_mean"] = float(np.mean(action))
        metrics[f"agent_{agent_idx}/action_abs_mean"] = float(np.mean(np.abs(action)))
        metrics[f"agent_{agent_idx}/action_clip_rate"] = float(np.mean(np.isclose(np.abs(action), 1.0, atol=1e-6)))
    return metrics


def _wandb_run_name(config):
    algorithm = config.get("algorithm", "unknown_algorithm")
    checkpoint_dir = config.get("paths", {}).get("checkpoint_dir")
    if checkpoint_dir:
        run_folder = os.path.basename(os.path.normpath(checkpoint_dir))
    else:
        run_folder = config.get("exp_name", "default_exp")
    return f"{algorithm}_{run_folder}"


def _init_wandb(config):
    if not config.get("flags", {}).get("use_wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "use_wandb=True but the wandb package is not installed. "
            "Install it with `pip install wandb` or add it from requirements.txt."
        ) from exc

    os.environ.setdefault("WANDB_DISABLE_GIT", "true")
    os.environ.setdefault("WANDB_CONSOLE", "off")
    wandb.init(
        project="MARL_2026_combine",
        name=_wandb_run_name(config),
        config=config,
        save_code=False,
    )
    return wandb


def _build_step_wandb_log(run, trainer, info, rewards, dones, total_steps, current_step):
    actions = _safe_float_array(run.get("pending_actions", []))
    rewards_arr = _safe_float_array(rewards)
    dones_arr = np.asarray(dones, dtype=bool)
    done_count = int(np.count_nonzero(dones_arr)) if dones_arr.size else 0
    check_goal = list(info.get("check_goal", []))
    collision_flags = list(info.get("bound_building_check", []))
    while len(collision_flags) < 4:
        collision_flags.append(False)

    noise_scale_fn = getattr(trainer, "_noise_scale", None)
    buffer_obj = getattr(trainer, "buffer", None)
    action_size = actions.size
    abs_actions = np.abs(actions) if action_size else actions

    metrics = {
        "train/total_step": total_steps,
        "train/episode_id": run["episode_id"],
        "train/episode_step": current_step,
        "train/episode_reward_running": run["episode_reward"],
        "train/reward_sum": float(np.sum(rewards_arr)) if rewards_arr.size else 0.0,
        "train/reward_mean": float(np.mean(rewards_arr)) if rewards_arr.size else 0.0,
        "train/reward_min": float(np.min(rewards_arr)) if rewards_arr.size else 0.0,
        "train/reward_max": float(np.max(rewards_arr)) if rewards_arr.size else 0.0,
        "train/done_any": int(done_count > 0),
        "train/done_count": done_count,
        "train/reached_count": int(sum(bool(value) for value in check_goal)),
        "train/reached_ratio": float(sum(bool(value) for value in check_goal)) / max(1, len(check_goal)),
        "train/noise_scale": noise_scale_fn() if callable(noise_scale_fn) else None,
        "train/buffer_size": len(buffer_obj) if buffer_obj is not None else None,
        "action/mean": float(np.mean(actions)) if action_size else 0.0,
        "action/std": float(np.std(actions)) if action_size else 0.0,
        "action/min": float(np.min(actions)) if action_size else 0.0,
        "action/max": float(np.max(actions)) if action_size else 0.0,
        "action/abs_mean": float(np.mean(abs_actions)) if action_size else 0.0,
        "action/clip_rate": float(np.mean(np.isclose(abs_actions, 1.0, atol=1e-6))) if action_size else 0.0,
        "env/collision_bound": int(bool(collision_flags[0])),
        "env/collision_building": int(bool(collision_flags[1])),
        "env/collision_drone": int(bool(collision_flags[2])),
        "env/collision_nearest_drone": int(bool(collision_flags[3])),
        "env/collision_any": int(any(bool(flag) for flag in collision_flags[:3])),
    }
    algorithm_name = str(getattr(trainer, "config", {}).get("algorithm", "algorithm")).replace("-", "_")
    metrics.update(_agent_step_wandb_metrics(actions, rewards_arr, dones_arr, check_goal))
    metrics.update(_status_holder_wandb_metrics(info))
    metrics.update(_prefixed_metrics(f"{algorithm_name}_action", getattr(trainer, "last_action_info", {})))
    metrics.update(_prefixed_metrics(f"{algorithm_name}_update", getattr(trainer, "last_update_info", {})))
    return metrics


def _build_episode_wandb_log(run, env_cfg, info, dones, current_step, episode_wall_clock, noise_value):
    collision_flags = list(info.get("bound_building_check", []))
    while len(collision_flags) < 4:
        collision_flags.append(False)
    goal_count = int(sum(bool(value) for value in run["episode_goal_found"]))
    end_reason = "unknown"
    if run["episode_decision"][0]:
        end_reason = "max_steps"
    elif run["episode_decision"][1]:
        end_reason = "collision"
    elif run["episode_decision"][2]:
        end_reason = "all_goals"

    return {
        "episode/id": run["episode_id"],
        "episode/reward": run["episode_reward"],
        "episode/length": current_step,
        "episode/wall_clock_sec": episode_wall_clock,
        "episode/end_reason": end_reason,
        "episode/collision": int(bool(True in dones and current_step < env_cfg["max_steps"])),
        "episode/all_steps_used": int(bool(run["episode_decision"][0])),
        "episode/all_goals": int(bool(run["episode_decision"][2])),
        "episode/reached_count": goal_count,
        "episode/reached_ratio": float(goal_count) / max(1, len(run["episode_goal_found"])),
        "episode/noise_scale": noise_value,
        "episode/collision_bound": int(bool(collision_flags[0])),
        "episode/collision_building": int(bool(collision_flags[1])),
        "episode/collision_drone": int(bool(collision_flags[2])),
        "episode/collision_nearest_drone": int(bool(collision_flags[3])),
    }


def _select_actions(trainer, env_slot, norm_cur_state, evaluate):
    if hasattr(trainer, "select_action_from_env"):
        if "env" not in env_slot:
            raise ValueError("The ORCA algorithm does not support subprocess/vectorized environments.")
        return trainer.select_action_from_env(env_slot["env"], evaluate=evaluate)
    return trainer.select_action(norm_cur_state, evaluate=evaluate)


def _final_checkpoint_info(stop_mode, episode, total_steps):
    if stop_mode == "step":
        return "step", int(total_steps)
    if stop_mode == "episode":
        return "ep", int(episode)
    raise ValueError("Unsupported stop_mode: {}. Expected 'step' or 'episode'.".format(stop_mode))


def _write_training_log_header(log_path, checkpoint_dir, checkpoint_kind, checkpoint_value, total_steps, total_wall_clock):
    with open(log_path, "w") as handle:
        handle.write("Training Summary\n")
        handle.write("checkpoint_dir: {}\n".format(checkpoint_dir))
        handle.write("latest_checkpoint: {}{}\n".format(checkpoint_kind, int(checkpoint_value)))
        handle.write("total_steps_used: {}\n".format(int(total_steps)))
        handle.write("total_wall_clock_sec: {:.2f}\n".format(float(total_wall_clock)))


def _append_evaluation_log(log_path, eval_summary):
    with open(log_path, "a") as handle:
        handle.write("\nEvaluation Summary\n")
        handle.write("Total collision: {}\n".format(int(eval_summary["total_collision"])))
        handle.write("Collision to bound: {}\n".format(int(eval_summary["collision_to_bound"])))
        handle.write("Collision to building: {}\n".format(int(eval_summary["collision_to_building"])))
        handle.write("Collision to drone: {}\n".format(int(eval_summary["collision_to_drone"])))
        handle.write("Destination reached: {}\n".format(int(eval_summary["destination_reached"])))
        handle.write("Idle UAV: {}\n".format(int(eval_summary["idle_uav"])))


def _append_evaluation_status(log_path, status, message=None):
    with open(log_path, "a") as handle:
        handle.write("\nEvaluation Status\n")
        handle.write("status: {}\n".format(status))
        if message:
            handle.write("message: {}\n".format(message))


def main(config):
    training_start_time = time.perf_counter()
    config = create_training_run_dirs(config)
    config["mode"] = "train"
    wandb_module = _init_wandb(config)
    env_cfg = config["env"]
    trainer = build_trainer(config)
    train_cfg = config["train"]
    checkpoint_dir = config.get("paths", {}).get("checkpoint_dir", "checkpoints")
    plot_dir = config.get("paths", {}).get("plot_dir")
    num_episodes = train_cfg["num_episodes"]
    total_steps_budget = train_cfg["total_steps"]
    num_parallel_envs = max(1, int(train_cfg.get("num_parallel_envs", 1)))
    stop_mode = train_cfg.get("stop_mode", "episode")
    if stop_mode not in ("episode", "step"):
        raise ValueError("Unsupported stop_mode: {}. Expected 'episode' or 'step'.".format(stop_mode))
    stop_target = num_episodes if stop_mode == "episode" else total_steps_budget
    stop_target_name = "episodes" if stop_mode == "episode" else "steps"
    env_template_count = min(num_parallel_envs, stop_target)
    if config.get("algorithm", "").lower() == "orca":
        env_template_count = 1
        num_parallel_envs = 1
    use_subproc_envs = env_template_count > 1
    if use_subproc_envs and env_template_count > 1:
        vec_env = SubprocVecEnv(config, env_template_count)
        plot_env_template = SharedMultiAgentEnv.from_config(config)
        env_slots = [
            {"vec_env": vec_env, "env_idx": env_idx, "n_agents": env_cfg["n_agents"]}
            for env_idx in range(env_template_count)
        ]
    else:
        vec_env = None
        plot_env_template = None
        env_slots = [{"env": SharedMultiAgentEnv.from_config(config)} for _ in range(env_template_count)]
    total_steps = 0
    episode = 0
    score_history = []
    eps_reward_record = []
    eps_noise_record = []
    eps_time_record = []
    eps_wall_clock_record = []
    eps_check_collision = []
    collision_count = 0
    drone_reached_per_eps = 0
    all_steps_used = 0
    crash_to_bound = 0
    crash_to_building = 0
    crash_to_drone = 0
    crash_due_to_nearest = 0
    active_runs = []
    stop_training = False

    print(f"[TRAIN] Checkpoints directory: {checkpoint_dir}")
    if plot_dir is not None:
        print(f"[TRAIN] Plot directory: {plot_dir}")
    print(f"[TRAIN] Parallel environments: {num_parallel_envs}")
    print(f"[TRAIN] Env execution mode: {'subprocess' if use_subproc_envs else 'serial'}")
    print(f"[TRAIN] Stop mode: {stop_mode} ({stop_target_name}={stop_target})")

    for env_slot in env_slots:
        if stop_mode == "episode" and episode >= num_episodes:
            break
        if stop_mode == "step" and total_steps >= total_steps_budget:
            break
        episode += 1
        active_runs.append(_start_episode_run(env_slot, trainer, episode, "subproc" if use_subproc_envs else "serial"))

    try:
        while active_runs and not stop_training:
            if stop_mode == "step" and total_steps >= total_steps_budget:
                break

            ready_runs = list(active_runs)
            if stop_mode == "step":
                remaining_steps = max(0, total_steps_budget - total_steps)
                ready_runs = ready_runs[:remaining_steps]
            if not ready_runs:
                break

            if use_subproc_envs:
                env_indices = []
                action_batch = []
                for run in ready_runs:
                    if hasattr(trainer, "begin_episode"):
                        trainer.begin_episode(run["episode_id"])
                    run["pending_actions"] = _select_actions(trainer, run["env_slot"], run["norm_cur_state"], evaluate=False)
                    env_indices.append(run["env_slot"]["env_idx"])
                    action_batch.append(run["pending_actions"])
                vec_env.step_async(env_indices, action_batch)  # .send() inside is non-blocking. dispatch actions to all environment workers without waiting for their results
                step_results = vec_env.step_wait(env_indices)
                run_results = list(zip(ready_runs, step_results))
            else:
                run_results = []
                for run in ready_runs:
                    if hasattr(trainer, "begin_episode"):
                        trainer.begin_episode(run["episode_id"])
                    actions = _select_actions(trainer, run["env_slot"], run["norm_cur_state"], evaluate=False)
                    env = run["env_slot"]["env"]
                    step_result = env.step(actions)
                    run["pending_actions"] = actions
                    run_results.append((run, step_result + (None,)))

            for run, step_result in run_results:
                if stop_mode == "step" and total_steps >= total_steps_budget:
                    stop_training = True
                    break

                current_step = len(run["trajectory_eachPlay"]) + 1
                if use_subproc_envs:
                    next_state_norm, next_state, rewards, dones, info, agent_snapshot = step_result
                    run["agent_snapshot"] = agent_snapshot
                else:
                    next_state_norm, next_state, rewards, dones, info, _ = step_result
                    run["agent_snapshot"] = None

                trainer.store_transition(run["norm_cur_state"], run["pending_actions"], rewards, next_state_norm, dones)
                _, _, run["single_eps_critic_cal_record"] = trainer.update(
                    i_episode=run["episode_id"],
                    total_step_count=total_steps,
                    single_eps_critic_cal_record=run["single_eps_critic_cal_record"],
                )

                run["cur_state"] = next_state
                run["norm_cur_state"] = next_state_norm
                total_steps += 1
                run["episode_reward"] += sum(rewards)
                budget_reached = stop_mode == "step" and total_steps >= total_steps_budget

                traj_step_list = []
                if run["agent_snapshot"] is not None:
                    for agent_info in run["agent_snapshot"]:
                        each_agent_idx = agent_info["agent_idx"]
                        traj_step_list.append([
                            agent_info["pos"][0],
                            agent_info["pos"][1],
                            np.array(info["step_reward_record"][each_agent_idx][1]),
                        ])
                else:
                    env = run["env_slot"]["env"]
                    for each_agent_idx, each_agent in env.all_uavs.items():
                        traj_step_list.append([
                            each_agent.pos[0],
                            each_agent.pos[1],
                            np.array(info["step_reward_record"][each_agent_idx][1]),
                        ])
                run["trajectory_eachPlay"].append(traj_step_list)
                _wandb_log(
                    wandb_module,
                    _build_step_wandb_log(run, trainer, info, rewards, dones, total_steps, current_step),
                    step=total_steps,
                )
                run.pop("pending_actions", None)

                episode_finished = _finalize_episode_run(run, env_cfg, info, dones, current_step)
                if budget_reached:
                    print(f"[TRAIN] Step budget reached: {total_steps}/{total_steps_budget}")
                    stop_training = True

                if not episode_finished:
                    if stop_training:
                        break
                    continue

                score_history.append(run["episode_reward"])
                eps_reward_record.append(run["episode_reward"])
                eps_time_record.append(current_step)
                episode_wall_clock = time.perf_counter() - run["episode_start_time"]
                eps_wall_clock_record.append(episode_wall_clock)

                if True in dones and current_step < env_cfg["max_steps"]:
                    collision_count = collision_count + 1
                    eps_check_collision.append(True)
                    if info["bound_building_check"][0]:
                        crash_to_bound = crash_to_bound + 1
                    elif info["bound_building_check"][1]:
                        crash_to_building = crash_to_building + 1
                    elif info["bound_building_check"][2]:
                        crash_to_drone = crash_to_drone + 1
                        if info["bound_building_check"][3]:
                            crash_due_to_nearest = crash_due_to_nearest + 1
                else:
                    all_steps_used = all_steps_used + 1
                    eps_check_collision.append(False)

                drone_reached_per_eps = drone_reached_per_eps + sum(run["episode_goal_found"])
                noise_scale = getattr(trainer, "_noise_scale", None)
                noise_value = noise_scale() if callable(noise_scale) else None
                eps_noise_record.append(noise_value)
                _wandb_log(
                    wandb_module,
                    _build_episode_wandb_log(
                        run,
                        env_cfg,
                        info,
                        dones,
                        current_step,
                        episode_wall_clock,
                        noise_value,
                    ),
                    step=total_steps,
                )

                if stop_mode == "step":
                    print(
                        f"[TRAIN] Episode {run['episode_id']} | "
                        f"Steps: {total_steps}/{total_steps_budget} | "
                        f"Reward: {run['episode_reward']:.3f} "
                    )
                else:
                    print(
                        f"[TRAIN] Episode {run['episode_id']}/{num_episodes} | "
                        f"Reward: {run['episode_reward']:.3f} "
                    )

                if run["episode_id"] % 100 == 0:
                    print("collision count for last 100 episode is {}, {}%".format(
                        collision_count, round(collision_count / 100 * 100, 2)
                    ))
                    print("Collision due to bound is {}".format(crash_to_bound))
                    print("Collision due to building is {}".format(crash_to_building))
                    print("Collision due to drone is {}, among them, caused by nearest drone is {}".format(
                        crash_to_drone, crash_due_to_nearest
                    ))
                    print("all steps used count is {}, {}%".format(
                        all_steps_used, round(all_steps_used / 100 * 100, 2)
                    ))
                    print("Reached-goal drones count is {}, avg {:.2f} per episode".format(
                        drone_reached_per_eps, drone_reached_per_eps / 100
                    ))
                    print("Reached-goal drone ratio is {}%".format(
                        round(drone_reached_per_eps / (100 * env_cfg["n_agents"]) * 100, 2)
                    ))
                    collision_count = 0
                    drone_reached_per_eps = 0
                    all_steps_used = 0
                    crash_to_bound = 0
                    crash_to_building = 0
                    crash_to_drone = 0
                    crash_due_to_nearest = 0

                if run["episode_id"] % config["save_interval"] == 0:
                    if plot_dir is not None and run["trajectory_eachPlay"]:
                        if plot_env_template is not None and run["initial_agent_snapshot"] is not None:
                            plot_env = _build_plot_env_from_snapshot(plot_env_template, run["initial_agent_snapshot"])
                            save_gif(plot_env, run["trajectory_eachPlay"], plot_dir, "train", run["episode_id"])
                        elif "env" in run["env_slot"]:
                            save_gif(run["env_slot"]["env"], run["trajectory_eachPlay"], plot_dir, "train", run["episode_id"])
                    if plot_dir is not None and not run["trajectory_eachPlay"]:
                        print("[TRAIN] Skipping GIF export because trajectory is empty.")
                    trainer.save(
                        checkpoint_dir,
                        episode=run["episode_id"],
                        step=total_steps,
                        stop_mode=stop_mode,
                    )

                active_runs.remove(run)
                if (
                    (stop_mode == "episode" and episode < num_episodes)
                    or (stop_mode == "step" and not stop_training and total_steps < total_steps_budget)
                ):
                    episode += 1
                    active_runs.append(_start_episode_run(run["env_slot"], trainer, episode, "subproc" if use_subproc_envs else "serial"))
    finally:
        if vec_env is not None:
            vec_env.close()
        if wandb_module is not None:
            wandb_module.finish()

    with open(os.path.join(plot_dir, "all_episode_reward.pickle"), "wb") as handle:
        pickle.dump(eps_reward_record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(plot_dir, "all_episode_noise.pickle"), "wb") as handle:
        pickle.dump(eps_noise_record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(plot_dir, "all_episode_time.pickle"), "wb") as handle:
        pickle.dump(eps_time_record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(plot_dir, "all_episode_wall_clock.pickle"), "wb") as handle:
        pickle.dump(eps_wall_clock_record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(plot_dir, "all_episode_collision.pickle"), "wb") as handle:
        pickle.dump(eps_check_collision, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(plot_dir, "GFG.csv"), "w", newline="") as f:
        write = csv.writer(f)
        write.writerows([score_history])

    trainer.save(checkpoint_dir, episode=episode, step=total_steps, stop_mode=stop_mode)
    checkpoint_kind, checkpoint_value = _final_checkpoint_info(stop_mode, episode, total_steps)
    total_wall_clock = time.perf_counter() - training_start_time
    log_path = os.path.join(checkpoint_dir, "training_eval_summary.txt")
    _write_training_log_header(
        log_path,
        checkpoint_dir,
        checkpoint_kind,
        checkpoint_value,
        total_steps,
        total_wall_clock,
    )
    _append_evaluation_status(log_path, "pending", "Post-training evaluation has started.")

    from evaluate import main as evaluate_main

    eval_config = deepcopy(config)
    eval_config["mode"] = "eval"
    eval_config["paths"]["checkpoint_dir"] = checkpoint_dir
    eval_config["paths"]["checkpoint_kind"] = checkpoint_kind
    eval_config["paths"]["checkpoint_value"] = checkpoint_value
    try:
        eval_summary = evaluate_main(eval_config)
    except Exception as exc:
        _append_evaluation_status(log_path, "failed", repr(exc))
        raise
    _append_evaluation_status(log_path, "completed")
    _append_evaluation_log(log_path, eval_summary)

    print(f"[TRAIN] Total wall-clock time: {total_wall_clock:.2f}s")
    print(f"[TRAIN] Finished. Checkpoints saved to: {checkpoint_dir}")
    print(f"[TRAIN] Training and evaluation summary saved to: {log_path}")
