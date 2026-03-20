from envs.shared_env import SharedMultiAgentEnv
from agents import build_trainer
from config.paths import create_training_run_dirs



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
    collision_count = 0
    one_drone_reach = 0
    two_drone_reach = 0
    three_drone_reach = 0
    four_drone_reach = 0
    five_drone_reach = 0
    six_drone_reach = 0
    seven_drone_reach = 0
    all_drone_reach = 0
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
            elif all(info["check_goal"]):
                episode_decision[2] = True
                print(
                    "All agents have reached their destinations at step {}, episode {} terminated.".format(
                        current_step - 1, episode
                    )
                )
            elif all([agent.reach_target for _, agent in env.all_uavs.items()]):
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

                if True in dones and current_step < env_cfg["max_steps"]:
                    collision_count = collision_count + 1
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

                if True in episode_goal_found:
                    num_true = sum(episode_goal_found)
                    if num_true == 1:
                        one_drone_reach = one_drone_reach + 1
                    elif num_true == 2:
                        two_drone_reach = two_drone_reach + 1
                    elif num_true == 3:
                        three_drone_reach = three_drone_reach + 1
                    elif num_true == 4:
                        four_drone_reach = four_drone_reach + 1
                    elif num_true == 5:
                        five_drone_reach = five_drone_reach + 1
                    elif num_true == 6:
                        six_drone_reach = six_drone_reach + 1
                    elif num_true == 7:
                        seven_drone_reach = seven_drone_reach + 1
                    else:
                        all_drone_reach = all_drone_reach + 1
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
            print("One goal reached count is {}, {}%".format(
                one_drone_reach, round(one_drone_reach / num_episodes * 100, 2)
            ))
            print("Two goal reached count is {}, {}%".format(
                two_drone_reach, round(two_drone_reach / num_episodes * 100, 2)
            ))
            print("Three goal reached count is {}, {}%".format(
                three_drone_reach, round(three_drone_reach / num_episodes * 100, 2)
            ))
            print("Four goal reached count is {}, {}%".format(
                four_drone_reach, round(four_drone_reach / num_episodes * 100, 2)
            ))
            print("Five goal reached count is {}, {}%".format(
                five_drone_reach, round(five_drone_reach / num_episodes * 100, 2)
            ))
            print("Six goal reached count is {}, {}%".format(
                six_drone_reach, round(six_drone_reach / num_episodes * 100, 2)
            ))
            print("Seven goal reached count is {}, {}%".format(
                seven_drone_reach, round(seven_drone_reach / num_episodes * 100, 2)
            ))
            print("All goal reached count is {}, {}%".format(
                all_drone_reach, round(all_drone_reach / num_episodes * 100, 2)
            ))

            collision_count = 0
            one_drone_reach = 0
            two_drone_reach = 0
            three_drone_reach = 0
            four_drone_reach = 0
            five_drone_reach = 0
            six_drone_reach = 0
            seven_drone_reach = 0
            all_drone_reach = 0
            all_steps_used = 0
            crash_to_bound = 0
            crash_to_building = 0
            crash_to_drone = 0
            crash_due_to_nearest = 0

        if episode % config["save_interval"] == 0:
            trainer.save(checkpoint_dir)

    trainer.save(checkpoint_dir)
    print(f"[TRAIN] Finished. Checkpoints saved to: {checkpoint_dir}")
