import numpy as np
import os
from envs.shared_env import SharedMultiAgentEnv
from agents import build_trainer
from utils.plotting_helper import *


def main(config):
    config["mode"] = "eval"
    env = SharedMultiAgentEnv.from_config(config)
    trainer = build_trainer(config)
    checkpoint_dir = config.get("paths", {}).get("checkpoint_dir", "checkpoints")
    trainer.load(checkpoint_dir)

    flags = config.get("flags", {})
    env_cfg = config["env"]
    eval_episodes = config["eval"]["episodes"]
    evaluation_by_episode = flags.get("evaluation_by_episode", False)
    get_evaluation_status = flags.get("get_evaluation_status", False)
    simply_view_evaluation = flags.get("simply_view_evaluation", False)

    total_agent_num = env.n_agents
    total_step = 0
    collision_count = 0
    drone_reached_per_eps = 0
    all_steps_used = 0
    sorties_reached = 0
    idle_drone = 0
    crash_to_bound = 0
    crash_to_building = 0
    crash_to_drone = 0
    crash_due_to_nearest = 0
    steps_before_collide = []
    evaluation_OD_repeatability = []

    for episode in range(1, eval_episodes + 1):
        if hasattr(trainer, "begin_episode"):
            trainer.begin_episode(episode)
        cur_state, norm_cur_state = env.reset(show=0)
        accum_reward = 0
        step = 0
        episode_decision = [False] * 3
        trajectory_eachPlay = []
        eps_all_ac_OD_goal = {
            agent_idx: [agent.goal, agent.ini_pos]
            for agent_idx, agent in env.all_uavs.items()
        }
        episode_goal_found = [False] * total_agent_num

        while True:
            actions = trainer.select_action(norm_cur_state, evaluate=True)
            norm_next_state, next_state, reward_aft_action, done_aft_action, info = env.step(actions)

            step += 1
            total_step += 1
            cur_state = next_state
            norm_cur_state = norm_next_state

            check_goal = info["check_goal"]
            step_reward_record = info["step_reward_record"]
            eps_status_holder = info["status_holder"]
            bound_building_check = info["bound_building_check"]
            agent_reach_target = [agent.reach_target for agent_idx, agent in env.all_uavs.items()]

            if list(check_goal) != agent_reach_target:
                raise ValueError(
                    "Goal status mismatch at episode {}, step {}: info['check_goal']={} vs reach_target={}".format(
                        episode, step, list(check_goal), agent_reach_target
                    )
                )

            traj_step_list = []
            for each_agent_idx, each_agent in env.all_uavs.items():
                traj_step_list.append([
                    each_agent.pos[0],
                    each_agent.pos[1],
                    np.array(step_reward_record[each_agent_idx][1]),
                    eps_status_holder[each_agent_idx],
                ])
            trajectory_eachPlay.append(traj_step_list)
            accum_reward = accum_reward + sum(reward_aft_action)

            for agentIdx, agent in env.all_uavs.items():
                print(
                    "drone {}, next WP is {}, deviation from ref line is {}, ref_line_reward is {}, "
                    "actual dist to goal is {}, dist_goal_reward is {}, velocity is {}, step {} reward is {}".format(
                        agentIdx,
                        agent.goal[-1],
                        eps_status_holder[agentIdx]["deviation_to_ref_line"],
                        eps_status_holder[agentIdx]["deviation_to_ref_line_reward"],
                        eps_status_holder[agentIdx]["Euclidean_dist_to_goal"],
                        eps_status_holder[agentIdx]["goal_leading_reward"],
                        eps_status_holder[agentIdx]["current_drone_speed"],
                        step,
                        reward_aft_action[agentIdx],
                    )
                )

            if get_evaluation_status:
                if simply_view_evaluation:
                    print("Static trajectory viewing is not available in the current project.")
                else:
                    print("GIF export is not available in the current project.")

            if env_cfg["max_steps"] < step:
                episode_decision[0] = True
                print(
                    "Agents stuck in some places, maximum step in one episode reached, current episode {} ends, all {} steps used".format(
                        episode, env_cfg["max_steps"]
                    )
                )
            elif True in done_aft_action:
                episode_decision[1] = True
                print(
                    "Some agent triggers termination condition like collision, current episode {} ends at step {}".format(
                        episode, step - 1
                    )
                )
            elif all(check_goal) and all(agent_reach_target):
                episode_decision[2] = True
                static_png_path = os.path.join(
                    config["paths"]["project_root"],
                    "resources",
                    "static_traj_episode_{}.png".format(episode),
                )
                view_static_traj_DWTD(
                    env=env,
                    trajectory_eachPlay=trajectory_eachPlay,
                    random_map_idx=0,
                    save_path=static_png_path,
                    max_time_step=len(trajectory_eachPlay),
                )
                print(
                    "All agents have reached their destinations at step {}, episode {} terminated.".format(
                        step - 1, episode
                    )
                )

            if True in episode_decision:
                for agent_idx, agent in env.all_uavs.items():
                    episode_goal_found[agent_idx] = agent.reach_target

                print("[Episode %05d] reward %6.4f " % (episode, accum_reward))

                if evaluation_by_episode:
                    if True in done_aft_action and step < env_cfg["max_steps"]:
                        collision_count = collision_count + 1
                        steps_before_collide.append(step)
                        if bound_building_check[0]:
                            crash_to_bound = crash_to_bound + 1
                        elif bound_building_check[1]:
                            crash_to_building = crash_to_building + 1
                        elif bound_building_check[2]:
                            crash_to_drone = crash_to_drone + 1
                            if bound_building_check[3]:
                                crash_due_to_nearest = crash_due_to_nearest + 1
                    else:
                        all_steps_used = all_steps_used + 1

                    drone_reached_per_eps = drone_reached_per_eps + sum(episode_goal_found)
                else:
                    for each_agent in env.all_uavs.values():
                        if each_agent.bound_collision:
                            collision_count = collision_count + 1
                            crash_to_bound = crash_to_bound + 1
                        elif each_agent.building_collision:
                            collision_count = collision_count + 1
                            crash_to_building = crash_to_building + 1
                        elif each_agent.drone_collision:
                            collision_count = collision_count + 1
                            crash_to_drone = crash_to_drone + 1
                        elif each_agent.reach_target:
                            sorties_reached = sorties_reached + 1
                        else:
                            idle_drone = idle_drone + 1
                break

        evaluation_OD_repeatability.append([eps_all_ac_OD_goal, trajectory_eachPlay])
        print("saving")

    if evaluation_by_episode:
        print("total collision count is {}, {}%".format(collision_count, round(collision_count / eval_episodes * 100, 2)))
        print("Collision due to bound is {}".format(crash_to_bound))
        print("Collision due to building is {}".format(crash_to_building))
        print("Collision due to drone is {}, among them, caused by any of previous two nearest drone is {}".format(
            crash_to_drone, crash_due_to_nearest
        ))
        print("all steps used count is {}, {}%".format(all_steps_used, round(all_steps_used / eval_episodes * 100, 2)))
        print("Reached-goal drones count is {}, avg {:.2f} per episode".format(
            drone_reached_per_eps, drone_reached_per_eps / eval_episodes
        ))
        print("Reached-goal drone ratio is {}%".format(
            round(drone_reached_per_eps / (eval_episodes * env.n_agents) * 100, 2)
        ))
    else:
        print("Total collision {}".format(collision_count))
        print("Collision to bound {}".format(crash_to_bound))
        print("Collision to building {}".format(crash_to_building))
        print("Collision to drone {}".format(crash_to_drone))
        print("Destination reached {}".format(sorties_reached))
        print("Idle UAV {}".format(idle_drone))
