import multiprocessing as mp
import random
from copy import deepcopy

import numpy as np

from envs.shared_env import SharedMultiAgentEnv

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _extract_agent_snapshot(env):
    snapshot = []
    for agent_idx, agent in env.all_uavs.items():
        snapshot.append({
            "agent_idx": agent_idx,
            "agent_name": agent.agent_name,
            "map_idx": env.current_random_map_idx,
            "pos": tuple(agent.pos.tolist()),
            "ini_pos": tuple(agent.ini_pos),
            "goal": [tuple(point) for point in agent.goal],
            "heading": float(agent.heading),
            "protectiveBound": float(agent.protectiveBound),
            "detectionRange": float(agent.detectionRange),
            "reach_target": bool(agent.reach_target),
        })
    return snapshot


def _set_worker_seed(base_seed, worker_idx):
    seed = int(base_seed) + int(worker_idx)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, TypeError):
            pass


def _worker(remote, parent_remote, config, worker_idx):
    parent_remote.close()
    _set_worker_seed(config.get("seed", 777), worker_idx)
    env = SharedMultiAgentEnv.from_config(config)
    try:
        while True:
            command, data = remote.recv()
            if command == "reset":
                cur_state, norm_cur_state = env.reset(data, show=0)
                remote.send((cur_state, norm_cur_state, _extract_agent_snapshot(env)))
            elif command == "step":
                next_state_norm, next_state, rewards, dones, info = env.step(data)
                remote.send((
                    next_state_norm,
                    next_state,
                    rewards,
                    dones,
                    info,
                    _extract_agent_snapshot(env),
                ))
            elif command == "close":
                remote.close()
                break
            else:
                raise ValueError("Unknown worker command: {}".format(command))
    finally:
        remote.close()


class SubprocVecEnv:
    def __init__(self, config, num_envs):
        self.num_envs = int(num_envs)
        self._ctx = mp.get_context("spawn")
        self._remotes = []
        self._processes = []

        for worker_idx in range(self.num_envs):
            parent_remote, child_remote = self._ctx.Pipe()
            process = self._ctx.Process(
                target=_worker,
                args=(child_remote, parent_remote, deepcopy(config), worker_idx),
                daemon=True,
            )
            process.start()
            child_remote.close()
            self._remotes.append(parent_remote)
            self._processes.append(process)

    def _format_worker_failure(self, env_idx):
        process = self._processes[env_idx]
        return "Subprocess environment {} terminated unexpectedly (exitcode={}).".format(
            env_idx,
            process.exitcode,
        )

    def reset_at(self, env_idx, episode):
        self._remotes[env_idx].send(("reset", episode))
        try:
            return self._remotes[env_idx].recv()
        except EOFError as exc:
            raise RuntimeError(self._format_worker_failure(env_idx)) from exc

    def step_at(self, env_idx, actions):
        self._remotes[env_idx].send(("step", actions))
        try:
            return self._remotes[env_idx].recv()
        except EOFError as exc:
            raise RuntimeError(self._format_worker_failure(env_idx)) from exc

    def step_async(self, env_indices, actions_batch):
        for env_idx, actions in zip(env_indices, actions_batch):
            self._remotes[env_idx].send(("step", actions))

    def step_wait(self, env_indices):
        results = []
        for env_idx in env_indices:
            try:
                results.append(self._remotes[env_idx].recv())
            except EOFError as exc:
                raise RuntimeError(self._format_worker_failure(env_idx)) from exc
        return results

    def close(self):
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self._processes:
            process.join(timeout=1.0)
