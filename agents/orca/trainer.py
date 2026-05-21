import importlib
import os
import sys
from pathlib import Path

import numpy as np

from agents.base_trainer import BaseTrainer


DEFAULT_ORCA_SOURCE_DIR = r"F:\githubClone\deepQ_learning_newVer\nf_dqn_v3_2_LSTM_Attention"


class ORCATrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.requires_episode_reset = True
        paths_cfg = config.get("paths", {})
        self.orca_source_dir = Path(paths_cfg.get("orca_code_dir") or DEFAULT_ORCA_SOURCE_DIR)
        self._simulator_cls = None
        self._vector_cls = None
        self._simulator = None
        self._simulator_env_id = None
        self.last_action_info = {}
        self.last_update_info = {
            "update_performed": False,
            "actor_updated": False,
            "update_step": 0,
            "buffer_size": 0,
            "learning_starts": 0,
            "batch_size": 0,
            "policy_delay": 1,
            "l2_reg": 0.0,
            "non_stationary_adam": False,
            "policy_noise": 0.0,
            "noise_clip": 0.0,
            "max_grad_norm": 0.0,
        }

    def _ensure_backend_loaded(self):
        if self._simulator_cls is not None and self._vector_cls is not None:
            return

        if not self.orca_source_dir.exists():
            raise FileNotFoundError("ORCA source directory does not exist: {}".format(self.orca_source_dir))

        orca_module_dir = self.orca_source_dir / "ORCA"
        source_dir_str = str(self.orca_source_dir)
        orca_dir_str = str(orca_module_dir)
        if source_dir_str not in sys.path:
            sys.path.insert(0, source_dir_str)
        if orca_dir_str not in sys.path:
            sys.path.insert(0, orca_dir_str)

        simulator_module = importlib.import_module("simulator")
        vector_module = importlib.import_module("vector")
        self._simulator_cls = simulator_module.Simulator
        self._vector_cls = vector_module.Vector2

    def _polygon_vertices(self, polygon):
        coords = list(polygon.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        return [self._vector_cls(float(x), float(y)) for x, y in coords]

    def _ensure_simulator(self, env):
        self._ensure_backend_loaded()

        env_id = id(env)
        if self._simulator is not None and self._simulator_env_id == env_id:
            return self._simulator

        simulator = self._simulator_cls(env.global_time, env.time_step)
        if env.world_map_2D_polyList is None:
            raise ValueError("ORCA trainer requires environment polygons to be initialized.")

        occupied_polygons = env.world_map_2D_polyList[0][0]
        for polygon in occupied_polygons:
            simulator.add_obstacle(self._polygon_vertices(polygon))
        simulator.process_obstacles()

        self._simulator = simulator
        self._simulator_env_id = env_id
        return simulator

    @staticmethod
    def _goal_vectors(uav):
        goal_vectors = [[float(uav.pos[0]), float(uav.pos[1])]]
        remaining_waypoints = uav.waypoints if uav.waypoints is not None else uav.goal
        if remaining_waypoints is None:
            return goal_vectors
        for point in remaining_waypoints:
            goal_vectors.append([float(point[0]), float(point[1])])
        return goal_vectors

    def select_action_from_env(self, env, evaluate=False):
        simulator = self._ensure_simulator(env)
        simulator.global_time_ = env.global_time
        simulator.agents_ = []

        for uav in env.all_uavs.values():
            initial_velocity = self._vector_cls(float(uav.vel[0]), float(uav.vel[1]))
            simulator.add_agent(
                self._vector_cls(float(uav.pos[0]), float(uav.pos[1])),
                self._goal_vectors(uav),
                float(uav.protectiveBound),
                initial_velocity,
                float(uav.maxSpeed),
            )

        for agent_idx in range(simulator.num_agents):
            simulator.set_pref_vel_newVer(agent_idx, env.grid_length)
        simulator.kd_tree_.build_agent_tree()
        for agent_idx in range(simulator.num_agents):
            simulator.agents_[agent_idx].compute_neighbors()
            simulator.agents_[agent_idx].compute_new_velocity(agent_idx)

        actions = []
        for agent_idx, uav in env.all_uavs.items():
            desired_velocity = np.array(
                [
                    float(simulator.agents_[agent_idx].new_velocity_.x),
                    float(simulator.agents_[agent_idx].new_velocity_.y),
                ],
                dtype=np.float64,
            )
            current_velocity = np.asarray(uav.vel, dtype=np.float64)
            accel = (desired_velocity - current_velocity) / float(env.time_step)
            normalized_action = np.clip(accel / float(env.acc_max), -1.0, 1.0)
            actions.append(normalized_action)
        actions_arr = np.asarray(actions, dtype=np.float64)
        self.last_action_info = {
            "raw_mean": float(np.mean(actions_arr)),
            "raw_std": float(np.std(actions_arr)),
            "raw_min": float(np.min(actions_arr)),
            "raw_max": float(np.max(actions_arr)),
            "noise_scale": 0.0,
            "sampled_noise_mean": 0.0,
            "sampled_noise_std": 0.0,
            "final_mean": float(np.mean(actions_arr)),
            "final_std": float(np.std(actions_arr)),
            "final_min": float(np.min(actions_arr)),
            "final_max": float(np.max(actions_arr)),
            "final_abs_mean": float(np.mean(np.abs(actions_arr))),
            "clip_rate": float(np.mean(np.isclose(np.abs(actions_arr), 1.0, atol=1e-6))),
        }
        return actions

    def select_action(self, obs, evaluate=False):
        raise NotImplementedError("ORCA trainer requires environment-backed action selection.")

    def begin_episode(self, episode):
        return None

    def store_transition(self, obs, actions, rewards, next_obs, dones, history=None, cur_hidden=None, next_hidden=None):
        return None

    def update(self, i_episode=None, total_step_count=None, single_eps_critic_cal_record=None):
        if single_eps_critic_cal_record is None:
            single_eps_critic_cal_record = []
        return None, None, single_eps_critic_cal_record

    def save(self, path, episode=None, step=None, stop_mode=None):
        os.makedirs(path, exist_ok=True)
        return None

    def load(self, path, checkpoint_tag=None):
        return None
