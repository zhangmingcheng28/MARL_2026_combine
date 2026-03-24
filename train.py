import csv
import os
import pickle
import numpy as np

from envs.shared_env import SharedMultiAgentEnv
from agents import build_trainer
from config.paths import create_training_run_dirs
from utils.plotting_helper import *


def main(config):
    config = create_training_run_dirs(config)
    config["mode"] = "train"
    env_cfg = config["env"]
    env = SharedMultiAgentEnv.from_config(config)  # initialization + world creation
    trainer = build_trainer(config)
    train_cfg = config["train"]
    checkpoint_dir = config.get("paths", {}).get("checkpoint_dir", "checkpoints")
    plot_dir = config.get("paths", {}).get("plot_dir")
    num_episodes = train_cfg["num_episodes"]
    total_steps_budget = train_cfg["total_steps"]
    stop_mode = train_cfg.get("stop_mode", "episode")
    total_steps = 0
    episode = 0
    score_history = []
    eps_reward_record = []
    eps_noise_record = []
    eps_time_record = []
    eps_check_collision = []
    collision_count = 0
    drone_reached_per_eps = 0
    all_steps_used = 0
    crash_to_bound = 0
    crash_to_building = 0
    crash_to_drone = 0
    crash_due_to_nearest = 0

    print(f"[TRAIN] Checkpoints directory: {checkpoint_dir}")
    if plot_dir is not None:
        print(f"[TRAIN] Plot directory: {plot_dir}")

    while True:
        if stop_mode == "episode" and episode >= num_episodes:
            break
        if stop_mode == "step" and total_steps >= total_steps_budget:
            break

        episode += 1
        single_eps_critic_cal_record = []
        if hasattr(trainer, "begin_episode"):
            trainer.begin_episode(episode)
        cur_state, norm_cur_state = env.reset(show=0)
        episode_reward = 0.0
        episode_decision = [False] * 3
        episode_goal_found = [False] * env.n_agents
        trajectory_eachPlay = []

        for step in range(env_cfg["max_steps"]):
            if stop_mode == "step" and total_steps >= total_steps_budget:
                break

            actions = trainer.select_action(norm_cur_state, evaluate=False)
            next_state_norm, next_state, rewards, dones, info = env.step(actions)

            trainer.store_transition(norm_cur_state, actions, rewards, next_state_norm, dones)
            _, _, single_eps_critic_cal_record = trainer.update(
                i_episode=episode,
                total_step_count=total_steps,
                single_eps_critic_cal_record=single_eps_critic_cal_record,
            )

            cur_state = next_state
            norm_cur_state = next_state_norm
            total_steps += 1
            episode_reward += sum(rewards)
            current_step = step + 1
            info_check_goal = list(info["check_goal"])
            agent_reach_target = [agent.reach_target for _, agent in env.all_uavs.items()]

            # if info_check_goal != agent_reach_target:
            #     raise ValueError(
            #         "Goal status mismatch at episode {}, step {}: info['check_goal']={} vs reach_target={}".format(
            #             episode, current_step, info_check_goal, agent_reach_target
            #         )
            #     )

            traj_step_list = []
            for each_agent_idx, each_agent in env.all_uavs.items():
                traj_step_list.append([
                    each_agent.pos[0],
                    each_agent.pos[1],
                    np.array(info["step_reward_record"][each_agent_idx][1]),
                ])
            trajectory_eachPlay.append(traj_step_list)

            if env_cfg["max_steps"] < current_step:
                episode_decision[0] = True
                print(
                    "Agents stuck in some places, maximum step in one episode reached, current episode {} ends, all {} steps used".format(
                        episode, env_cfg["max_steps"]
                    )
                )
            elif True in dones:
                episode_decision[1] = True
                print(
                    "Some agent triggers termination condition like collision, current episode {} ends at step {}".format(
                        episode, current_step - 1
                    )
                )
            elif all(info_check_goal) or all(agent_reach_target):
            # elif all(info_check_goal) and all(agent_reach_target):
                episode_decision[2] = True
                print(
                    "All agents have reached their destinations at step {}, episode {} terminated.".format(
                        current_step - 1, episode
                    )
                )

            if True in episode_decision:
                for agent_idx, agent in env.all_uavs.items():
                    episode_goal_found[agent_idx] = agent.reach_target
                score_history.append(episode_reward)
                eps_reward_record.append(episode_reward)
                eps_time_record.append(current_step)

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

                drone_reached_per_eps = drone_reached_per_eps + sum(episode_goal_found)
                noise_scale = getattr(trainer, "_noise_scale", None)
                eps_noise_record.append(noise_scale() if callable(noise_scale) else None)
                break

        print(f"[TRAIN] Episode {episode}/{num_episodes} | Reward: {episode_reward:.3f}")

        if episode % 100 == 0:
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
                round(drone_reached_per_eps / (100 * env.n_agents) * 100, 2)
            ))
            save_gif(env, trajectory_eachPlay, plot_dir, "train", episode)
            collision_count = 0
            drone_reached_per_eps = 0
            all_steps_used = 0
            crash_to_bound = 0
            crash_to_building = 0
            crash_to_drone = 0
            crash_due_to_nearest = 0

        if episode % config["save_interval"] == 0:
            trainer.save(checkpoint_dir, episode=episode, step=total_steps)

    with open(os.path.join(plot_dir, "all_episode_reward.pickle"), "wb") as handle:
        pickle.dump(eps_reward_record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(plot_dir, "all_episode_noise.pickle"), "wb") as handle:
        pickle.dump(eps_noise_record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(plot_dir, "all_episode_time.pickle"), "wb") as handle:
        pickle.dump(eps_time_record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(plot_dir, "all_episode_collision.pickle"), "wb") as handle:
        pickle.dump(eps_check_collision, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(plot_dir, "GFG.csv"), "w", newline="") as f:
        write = csv.writer(f)
        write.writerows([score_history])

    trainer.save(checkpoint_dir, episode=episode, step=total_steps)
    print(f"[TRAIN] Finished. Checkpoints saved to: {checkpoint_dir}")
