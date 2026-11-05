"""Environment construction and vectorized-env wrappers."""

import gymnasium as gym
import numpy as np
import torch


def _limit_cv2_threads():
    try:
        import cv2
    except ImportError:
        return
    cv2.setNumThreads(0)


class _EnvFactory:
    """Picklable env thunk; subclasses build the wrapper stack in ``__call__``."""

    def __init__(self, env_id, idx, capture_video, run_name):
        self.env_id = env_id
        self.idx = idx
        self.capture_video = capture_video
        self.run_name = run_name

    def _make(self, **kwargs):
        if self.capture_video and self.idx == 0:
            env = gym.make(self.env_id, render_mode="rgb_array", **kwargs)
            return gym.wrappers.RecordVideo(env, f"videos/{self.run_name}")
        return gym.make(self.env_id, **kwargs)


class _AtariEnvFactory(_EnvFactory):
    def __call__(self):
        from stable_baselines3.common.atari_wrappers import (
            ClipRewardEnv,
            EpisodicLifeEnv,
            FireResetEnv,
            MaxAndSkipEnv,
            NoopResetEnv,
        )

        _limit_cv2_threads()
        env = self._make()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env


def make_env_atari(env_id, idx, capture_video, run_name):
    return _AtariEnvFactory(env_id, idx, capture_video, run_name)


def _clip_obs(obs):
    return np.clip(obs, -10, 10)


def _clip_reward(reward):
    return np.clip(reward, -10, 10)


class _MuJoCoEnvFactory(_EnvFactory):
    def __init__(self, env_id, idx, capture_video, run_name, gamma):
        super().__init__(env_id, idx, capture_video, run_name)
        self.gamma = gamma

    def __call__(self):
        env = self._make()
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            _clip_obs,
            observation_space=env.observation_space,
        )
        env = gym.wrappers.NormalizeReward(env, gamma=self.gamma)
        env = gym.wrappers.TransformReward(env, _clip_reward)
        return env


def make_env_mujoco(env_id, idx, capture_video, run_name, gamma):
    return _MuJoCoEnvFactory(env_id, idx, capture_video, run_name, gamma)


# ProcGen per-game episode step limits, from procgen/src/games/*.cpp
_PROCGEN_STEP_LIMITS = {
    "bigfish": 6000,
    "bossfight": 4000,
    "caveflyer": 1000,
    "chaser": 1000,
    "climber": 1000,
    "coinrun": 1000,
    "dodgeball": 1000,
    "fruitbot": 1000,
    "heist": 1000,
    "jumper": 1000,
    "leaper": 500,
    "maze": 500,
    "miner": 1000,
    "ninja": 1000,
    "plunder": 4000,
    "starpilot": 1000,
}


class ProcGenVecEnv:
    """ProcgenEnv behind a gymnasium vector env interface, with reward normalization and clipping."""

    def __init__(
        self,
        env_name,
        num_envs,
        gamma=0.999,
        num_levels=0,
        start_level=0,
        distribution_mode="easy",
    ):
        from procgen import ProcgenEnv as _ProcgenEnv

        self._env = _ProcgenEnv(
            num_envs=num_envs,
            env_name=env_name,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode=distribution_mode,
        )
        self.num_envs = num_envs
        self._step_limit = _PROCGEN_STEP_LIMITS.get(env_name, 1000)

        self.single_observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, 84, 84),
            dtype=np.uint8,
        )
        self.single_action_space = gym.spaces.Discrete(self._env.action_space.n)
        self.is_vector_env = True

        self._ep_returns = np.zeros(num_envs, dtype=np.float64)
        self._ep_lengths = np.zeros(num_envs, dtype=np.int32)

        # Stand-in for the terminal frame on truncation, which ProcGen's auto-reset discards.
        self._last_obs_processed = None

        self._gamma = gamma
        self._ret = np.zeros(num_envs, dtype=np.float64)
        self._rew_mean = 0.0
        self._rew_var = 1.0
        self._rew_count = 1e-4

    def _process_obs(self, obs_dict):
        import cv2

        rgb = obs_dict["rgb"]
        out = np.empty((rgb.shape[0], 3, 84, 84), dtype=np.uint8)
        for i in range(rgb.shape[0]):
            out[i] = cv2.resize(
                rgb[i], (84, 84), interpolation=cv2.INTER_AREA
            ).transpose(2, 0, 1)
        return out

    def _normalize_reward(self, reward):
        self._ret = self._ret * self._gamma + reward
        batch_mean = np.mean(self._ret)
        batch_var = np.var(self._ret)
        batch_count = len(self._ret)
        delta = batch_mean - self._rew_mean
        total = self._rew_count + batch_count
        self._rew_mean += delta * batch_count / total
        m_a = self._rew_var * self._rew_count
        m_b = batch_var * batch_count
        self._rew_var = (
            m_a + m_b + delta**2 * self._rew_count * batch_count / total
        ) / total
        self._rew_count = total
        return reward / (np.sqrt(self._rew_var) + 1e-8)

    def reset(self, seed=None, options=None):
        obs = self._env.reset()
        self._ep_returns[:] = 0
        self._ep_lengths[:] = 0
        self._ret[:] = 0
        processed = self._process_obs(obs)
        self._last_obs_processed = processed.copy()
        return processed, {}

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        obs, reward, done, info = self._env.step(action)
        reward = reward.astype(np.float64)

        self._ep_returns += reward
        self._ep_lengths += 1

        # ProcGen does not flag timeouts, so an episode reaching the step limit counts as truncated.
        done_bool = done.astype(bool)
        truncated = done_bool & (self._ep_lengths >= self._step_limit)
        terminated = done_bool & ~truncated

        final_info = [None] * self.num_envs
        final_observation = [None] * self.num_envs
        for i in range(self.num_envs):
            if done[i]:
                final_info[i] = {
                    "episode": {
                        "r": np.float32(self._ep_returns[i]),
                        "l": int(self._ep_lengths[i]),
                    }
                }
                self._ep_returns[i] = 0
                self._ep_lengths[i] = 0
            if truncated[i] and self._last_obs_processed is not None:
                final_observation[i] = self._last_obs_processed[i].copy()

        reward = self._normalize_reward(reward)
        reward = np.clip(reward, -10, 10).astype(np.float32)
        self._ret[done_bool] = 0

        processed = self._process_obs(obs)
        self._last_obs_processed = processed.copy()

        infos = {
            "final_info": final_info,
            "final_observation": final_observation,
        }
        return processed, reward, terminated, truncated, infos

    def close(self):
        self._env.close()

    def _gym3_env(self):
        env = self._env
        while not hasattr(env, "callmethod") and hasattr(env, "env"):
            env = env.env
        if not hasattr(env, "callmethod"):
            raise AttributeError(
                "Could not find gym3 env with callmethod on ProcgenEnv chain"
            )
        return env

    def get_state(self):
        return {
            "env_states": self._gym3_env().callmethod("get_state"),
            "ep_returns": self._ep_returns.copy(),
            "ep_lengths": self._ep_lengths.copy(),
            "ret": self._ret.copy(),
            "rew_mean": self._rew_mean,
            "rew_var": self._rew_var,
            "rew_count": self._rew_count,
            "last_obs_processed": (
                None
                if self._last_obs_processed is None
                else self._last_obs_processed.copy()
            ),
        }

    def set_state(self, state):
        self._gym3_env().callmethod("set_state", state["env_states"])
        self._ep_returns = state["ep_returns"].copy()
        self._ep_lengths = state["ep_lengths"].copy()
        self._ret = state["ret"].copy()
        self._rew_mean = state["rew_mean"]
        self._rew_var = state["rew_var"]
        self._rew_count = state["rew_count"]
        last_obs = state.get("last_obs_processed")
        self._last_obs_processed = None if last_obs is None else last_obs.copy()


def force_real_reset(env):
    """Reset an Atari env to a new game rather than the next life."""
    from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv

    inner = env
    while inner is not None:
        if isinstance(inner, EpisodicLifeEnv):
            inner.was_real_done = True
            break
        inner = getattr(inner, "env", None)
    return env.reset()


def classify_env(env_id):
    return {
        "is_atari": "NoFrameskip" in env_id,
        "is_procgen": "procgen" in env_id,
    }


def build_vec_env_fn(args, run_name):
    kinds = classify_env(args.env_id)

    if kinds["is_atari"]:

        def vec_env_fn(n_envs: int = args.num_envs):
            return gym.vector.SyncVectorEnv(
                [
                    make_env_atari(args.env_id, i, args.capture_video, run_name)
                    for i in range(n_envs)
                ],
            )
    elif kinds["is_procgen"]:
        procgen_game = args.env_id.split("-", 1)[1]

        def vec_env_fn(n_envs: int = args.num_envs):
            return ProcGenVecEnv(
                env_name=procgen_game,
                num_envs=n_envs,
                gamma=args.gamma,
                num_levels=args.num_levels,
                start_level=args.start_level,
                distribution_mode=args.distribution_mode,
            )
    else:

        def vec_env_fn(n_envs: int = args.num_envs):
            return gym.vector.SyncVectorEnv(
                [
                    make_env_mujoco(
                        args.env_id, i, args.capture_video, run_name, args.gamma
                    )
                    for i in range(n_envs)
                ],
                autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
            )

    return vec_env_fn
