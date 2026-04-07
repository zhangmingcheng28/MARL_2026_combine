import multiprocessing as mp
from copy import deepcopy

from envs.shared_env import SharedMultiAgentEnv


def _extract_agent_snapshot(env):
    snapshot = []
    for agent_idx, agent in env.all_uavs.items():
        snapshot.append({
            "agent_idx": agent_idx,
            "agent_name": agent.agent_name,
            "pos": tuple(agent.pos.tolist()),
            "ini_pos": tuple(agent.ini_pos),
            "goal": [tuple(point) for point in agent.goal],
            "heading": float(agent.heading),
            "protectiveBound": float(agent.protectiveBound),
            "detectionRange": float(agent.detectionRange),
            "reach_target": bool(agent.reach_target),
        })
    return snapshot


def _worker(remote, parent_remote, config):
    parent_remote.close()
    env = SharedMultiAgentEnv.from_config(config)
    try:
        while True:
            command, data = remote.recv()
            if command == "reset":
                cur_state, norm_cur_state = env.reset(show=0)
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

        for _ in range(self.num_envs):
            parent_remote, child_remote = self._ctx.Pipe()
            process = self._ctx.Process(
                target=_worker,
                args=(child_remote, parent_remote, deepcopy(config)),
                daemon=True,
            )
            process.start()
            child_remote.close()
            self._remotes.append(parent_remote)
            self._processes.append(process)

    def reset_at(self, env_idx):
        self._remotes[env_idx].send(("reset", None))
        return self._remotes[env_idx].recv()

    def step_at(self, env_idx, actions):
        self._remotes[env_idx].send(("step", actions))
        return self._remotes[env_idx].recv()

    def close(self):
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self._processes:
            process.join(timeout=1.0)
