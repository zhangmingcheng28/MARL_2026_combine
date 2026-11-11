"""Legacy no-CR waypoint tracking, adapted to normalized environment actions."""

import numpy as np

from agents.base_trainer import BaseTrainer


class NoCRTrainer(BaseTrainer):
    requires_checkpoint = False

    def __init__(self, config):
        super().__init__(config)
        if config.get("mode") == "train":
            raise ValueError("no-cr is an evaluation-only controller; use --mode evaluate.")
        self.numpy_dtype = np.float64 if config.get("dtype") == "float64" else np.float32
        self.last_action_info = {}
        self.last_update_info = {"update_performed": False, "actor_updated": False}
        self.begin_episode(0)

    def begin_episode(self, episode):
        self._waypoint_indices = {}
        self._env = None
        self._last_step = -1

    def select_action_from_env(self, env, evaluate=False):
        if env.acc_max <= 0 or env.time_step <= 0:
            raise ValueError("no-cr requires positive acc_max and time_step.")
        if self._env is not env or env.step_count < self._last_step:
            self.begin_episode(0)
            self._env = env
        new_step = env.step_count != self._last_step
        actions = []
        for agent_idx, uav in env.all_uavs.items():
            if uav.reach_target or uav.bound_collision or uav.building_collision or uav.drone_collision:
                actions.append(np.zeros(2, dtype=self.numpy_dtype))
                continue
            route = uav.goal
            if route is None or len(route) == 0:
                raise ValueError("no-cr requires a planned route for every active UAV.")
            waypoint_index = self._waypoint_indices.get(agent_idx, 0)
            # Legacy code advances at most one waypoint after a step when
            # two circles of radius protectiveBound overlap. Final arrival
            # remains the shared environment's decision for all algorithms.
            if new_step and env.step_count > 0 and waypoint_index < len(route) - 1:
                distance = np.linalg.norm(np.asarray(route[waypoint_index]) - uav.pos)
                if distance <= 2.0 * uav.protectiveBound:
                    waypoint_index += 1
            self._waypoint_indices[agent_idx] = waypoint_index
            direction = np.asarray(route[waypoint_index], dtype=np.float64) - uav.pos
            distance = np.linalg.norm(direction)
            desired_velocity = direction * (uav.maxSpeed / distance) if distance > 0 else np.zeros(2)
            acceleration = (desired_velocity - uav.vel) / env.time_step
            magnitude = np.linalg.norm(acceleration)
            if magnitude > env.acc_max:
                acceleration *= env.acc_max / magnitude
            # SharedMultiAgentEnv.step multiplies actions by acc_max, unlike
            # the legacy no_cr branch, which accepts physical acceleration.
            actions.append(np.clip(acceleration / env.acc_max, -1.0, 1.0).astype(self.numpy_dtype))
        self._last_step = env.step_count
        action_array = np.asarray(actions)
        self.last_action_info = {
            "noise_scale": 0.0,
            "final_mean": float(action_array.mean()),
            "final_std": float(action_array.std()),
            "final_min": float(action_array.min()),
            "final_max": float(action_array.max()),
        }
        return actions

    def select_action(self, obs, evaluate=False):
        raise NotImplementedError("no-cr requires environment-backed action selection.")

    def store_transition(self, obs, actions, rewards, next_obs, dones):
        return None

    def update(self, i_episode=None, total_step_count=None, single_eps_critic_cal_record=None):
        return None, None, [] if single_eps_critic_cal_record is None else single_eps_critic_cal_record

    def save(self, path, episode=None, step=None, stop_mode=None):
        return None

    def load(self, path, checkpoint_tag=None):
        return None
