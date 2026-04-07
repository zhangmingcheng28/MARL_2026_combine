import csv
import os
import pickle
import time
from types import SimpleNamespace
import numpy as np

from envs.shared_env import SharedMultiAgentEnv
from envs.vector_env import SubprocVecEnv
from agents import build_trainer
from config.paths import create_training_run_dirs
from utils.plotting_helper import *


def _build_plot_env_from_snapshot(template_env, agent_snapshot):
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
        cur_state, norm_cur_state, agent_snapshot = env_slot["vec_env"].reset_at(env_slot["env_idx"])
        n_agents = env_slot["n_agents"]
    else:
        env = env_slot["env"]
        cur_state, norm_cur_state = env.reset(show=0)
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


def main(config):
    training_start_time = time.perf_counter()
    config = create_training_run_dirs(config)
    config["mode"] = "train"
    env_cfg = config["env"]
    trainer = build_trainer(config)
    train_cfg = config["train"]
    checkpoint_dir = config.get("paths", {}).get("checkpoint_dir", "checkpoints")
    plot_dir = config.get("paths", {}).get("plot_dir")
    num_episodes = train_cfg["num_episodes"]
    total_steps_budget = train_cfg["total_steps"]
    num_parallel_envs = max(1, int(train_cfg.get("num_parallel_envs", 1)))
    use_subproc_envs = num_parallel_envs > 1
    stop_mode = train_cfg.get("stop_mode", "episode")
    env_template_count = min(num_parallel_envs, num_episodes)
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

    print(f"[TRAIN] Checkpoints directory: {checkpoint_dir}")
    if plot_dir is not None:
        print(f"[TRAIN] Plot directory: {plot_dir}")
    print(f"[TRAIN] Parallel environments: {num_parallel_envs}")
    print(f"[TRAIN] Env execution mode: {'subprocess' if use_subproc_envs else 'serial'}")

    for env_slot in env_slots:
        if episode >= num_episodes:
            break
        episode += 1
        active_runs.append(_start_episode_run(env_slot, trainer, episode, "subproc" if use_subproc_envs else "serial"))

    try:
        while active_runs:
            if stop_mode == "episode" and episode >= num_episodes and not active_runs:
                break
            if stop_mode == "step" and total_steps >= total_steps_budget:
                break

            for run in list(active_runs):
                if stop_mode == "step" and total_steps >= total_steps_budget:
                    break

                current_step = len(run["trajectory_eachPlay"]) + 1

                if hasattr(trainer, "begin_episode"):
                    trainer.begin_episode(run["episode_id"])

                actions = trainer.select_action(run["norm_cur_state"], evaluate=False)
                if use_subproc_envs and "vec_env" in run["env_slot"]:
                    next_state_norm, next_state, rewards, dones, info, agent_snapshot = run["env_slot"]["vec_env"].step_at(
                        run["env_slot"]["env_idx"],
                        actions,
                    )
                    run["agent_snapshot"] = agent_snapshot
                else:
                    env = run["env_slot"]["env"]
                    next_state_norm, next_state, rewards, dones, info = env.step(actions)
                    run["agent_snapshot"] = None

                trainer.store_transition(run["norm_cur_state"], actions, rewards, next_state_norm, dones)
                _, _, run["single_eps_critic_cal_record"] = trainer.update(
                    i_episode=run["episode_id"],
                    total_step_count=total_steps,
                    single_eps_critic_cal_record=run["single_eps_critic_cal_record"],
                )

                run["cur_state"] = next_state
                run["norm_cur_state"] = next_state_norm
                total_steps += 1
                run["episode_reward"] += sum(rewards)

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

                if not _finalize_episode_run(run, env_cfg, info, dones, current_step):
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
                eps_noise_record.append(noise_scale() if callable(noise_scale) else None)

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
                    if plot_dir is not None and run["trajectory_eachPlay"]:
                        if plot_env_template is not None and run["initial_agent_snapshot"] is not None:
                            plot_env = _build_plot_env_from_snapshot(plot_env_template, run["initial_agent_snapshot"])
                            save_gif(plot_env, run["trajectory_eachPlay"], plot_dir, "train", run["episode_id"])
                        elif "env" in run["env_slot"]:
                            save_gif(run["env_slot"]["env"], run["trajectory_eachPlay"], plot_dir, "train", run["episode_id"])
                    if plot_dir is not None and not run["trajectory_eachPlay"]:
                        print("[TRAIN] Skipping GIF export because trajectory is empty.")
                    collision_count = 0
                    drone_reached_per_eps = 0
                    all_steps_used = 0
                    crash_to_bound = 0
                    crash_to_building = 0
                    crash_to_drone = 0
                    crash_due_to_nearest = 0

                if run["episode_id"] % config["save_interval"] == 0:
                    trainer.save(checkpoint_dir, episode=run["episode_id"], step=total_steps)

                active_runs.remove(run)
                if episode < num_episodes and not (stop_mode == "step" and total_steps >= total_steps_budget):
                    episode += 1
                    active_runs.append(_start_episode_run(run["env_slot"], trainer, episode, "subproc" if use_subproc_envs else "serial"))
    finally:
        if vec_env is not None:
            vec_env.close()

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

    trainer.save(checkpoint_dir, episode=episode, step=total_steps)
    total_wall_clock = time.perf_counter() - training_start_time
    print(f"[TRAIN] Total wall-clock time: {total_wall_clock:.2f}s")
    print(f"[TRAIN] Finished. Checkpoints saved to: {checkpoint_dir}")
