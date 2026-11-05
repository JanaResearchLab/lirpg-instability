from itertools import chain

import gymnasium as gym
import numpy as np
import torch
from torch import nn


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def get_cnn_model(input_dim):
    return nn.Sequential(
        layer_init(nn.Conv2d(input_dim, 32, kernel_size=8, stride=4)),
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        layer_init(nn.Linear(64 * 7 * 7, 512)),
        nn.ReLU(),
    )


def get_linear_model(input_dim, hidden_dim=64):
    return nn.Sequential(
        layer_init(nn.Linear(input_dim, hidden_dim)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden_dim, hidden_dim)),
        nn.Tanh(),
    )


class IntrinsicRewardFn(nn.Module):
    """The learned intrinsic reward r_in and the extrinsic value function V_ex used by its meta-update."""

    def __init__(
        self,
        envs,
        shared_rin_vex: bool = False,
        r_in_per_action: bool = False,
        activation_fn: str = "tanh",
        use_mode_flag: bool = False,
    ):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        obs_dim = np.array(obs_shape).prod()

        if r_in_per_action:
            assert isinstance(envs.single_action_space, gym.spaces.Discrete), (
                "r_in_per_action is only supported for discrete action spaces"
            )
            r_in_out_dim = envs.single_action_space.n
            print(
                f"Initializing Intrinsic Reward Function with output dimension: {r_in_out_dim}"
            )
        else:
            r_in_out_dim = 1
            print("Initializing Intrinsic Reward Function with output dimension: 1")

        if len(obs_shape) == 1:
            print("Initializing Intrinsic Reward Function for Vector Inputs")
            if shared_rin_vex:
                input_dim = obs_dim
                output_dim = hidden_dim = 64
                base_model = get_linear_model(input_dim, hidden_dim)
                value_model = base_model
            else:
                # r_in(s, a) takes the state and action concatenated.
                assert isinstance(envs.single_action_space, gym.spaces.Box)
                input_dim = obs_dim + envs.single_action_space.shape[0]
                output_dim = hidden_dim = 64
                print("Initializing Intrinsic Reward Function for MuJoCo")
                base_model = get_linear_model(input_dim, hidden_dim)
                value_model = get_linear_model(obs_dim, hidden_dim)
            self.norm_frac = 1.0
        else:
            print("Initializing Intrinsic Reward Function for Atari")
            input_dim = obs_shape[0]
            base_model = get_cnn_model(input_dim)
            if shared_rin_vex:
                value_model = base_model
            else:
                value_model = get_cnn_model(input_dim)
            output_dim = 512
            self.norm_frac = 1.0 / 255

        self.base_model = base_model
        self.value_model = value_model

        if activation_fn == "tanh":
            activation = nn.Tanh()
        elif activation_fn == "relu":
            activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")
        self.r_in = nn.Sequential(
            base_model, nn.Linear(output_dim, r_in_out_dim), activation
        )
        self.v_ex = nn.Sequential(value_model, nn.Linear(output_dim, 1))

        self.use_mode_flag = use_mode_flag
        if use_mode_flag:
            self.mode_embedding = nn.Linear(2, output_dim)

    def _mode_onehot(self, mode_flag):
        return torch.nn.functional.one_hot(mode_flag.long(), 2).float()

    def get_meta_params(self, include_v_ex: bool = True):
        params = self.r_in.parameters()
        if include_v_ex:
            params = chain(params, self.v_ex.parameters())
        if self.use_mode_flag:
            params = chain(params, self.mode_embedding.parameters())
        return params

    def get_intrinsic_reward(self, x, actions=None, mode_flag=None):
        # `actions` is unused here; the caller handles state-action inputs.
        x = x * self.norm_frac

        if self.use_mode_flag and mode_flag is not None:
            features = self.base_model(x)
            features = features + self.mode_embedding(self._mode_onehot(mode_flag))
            for i in range(1, len(self.r_in)):
                features = self.r_in[i](features)
            return features

        return self.r_in(x)

    def get_extrinsic_value(self, x, mode_flag=None):
        x = x * self.norm_frac
        if self.use_mode_flag and mode_flag is not None:
            features = self.v_ex[0](x)
            features = features + self.mode_embedding(self._mode_onehot(mode_flag))
            for i in range(1, len(self.v_ex)):
                features = self.v_ex[i](features)
            return features
        return self.v_ex(x)


# From OpenAI Baselines (MIT, Copyright (c) 2017 OpenAI; see licenses/baselines-LICENSE.txt), as used in
# https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/mpi_util.py
class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon

    def update(self, x):
        x = x.detach().cpu()
        batch_mean, batch_std, batch_count = (
            torch.mean(x, dim=0),
            torch.std(x, dim=0),
            x.shape[0],
        )
        batch_var = torch.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RNDRewardFn(nn.Module):
    """Random Network Distillation bonus (Burda et al. 2018)."""

    def __init__(self, envs, obs_mean, obs_std, gamma):
        super().__init__()
        self.envs = envs
        obs_shape = envs.single_observation_space.shape

        self.obs_rms = RunningMeanStd(shape=obs_mean.shape)
        self.obs_rms.mean = torch.FloatTensor(obs_mean)
        self.obs_rms.var = torch.FloatTensor(obs_std).square()

        self.rew_rms = RunningMeanStd()
        self.discounted_reward = RewardForwardFilter(gamma)

        if isinstance(envs.single_action_space, gym.spaces.Box):
            print("Initializing RND for MuJoCo (vector inputs)")
            obs_dim = np.prod(obs_shape)
            self.input_type = "vector"

            self.target = get_linear_model(obs_dim)
            self.predictor = get_linear_model(obs_dim)
            output_dim = 64
        else:
            print("Initializing RND for Atari (image inputs)")
            obs_channels = obs_shape[0]
            self.input_type = "image"

            self.target = get_cnn_model(obs_channels)
            self.predictor = get_cnn_model(obs_channels)

            output_dim = 512

        self.target = nn.Sequential(self.target, nn.Linear(output_dim, 64))
        self.predictor = nn.Sequential(
            self.predictor,
            nn.Linear(output_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 64),
        )

        for param in self.target.parameters():
            param.requires_grad = False

    def get_intrinsic_reward(self, x, update_rms=True):
        if update_rms:
            self.obs_rms.update(x)
        orig_shape = x.shape
        x = x.reshape(-1, *self.obs_rms.mean.shape)
        x = torch.clamp(
            (x - self.obs_rms.mean.to(x.device))
            / (self.obs_rms.var.to(x.device).sqrt() + 1e-8),
            -5,
            5,
        )
        x = x.reshape(orig_shape)

        with torch.no_grad():
            target_feat = self.target(x)

        pred_feat = self.predictor(x)
        mse = (pred_feat - target_feat).pow(2).mean(dim=1)

        if update_rms:
            total_int_reward = mse.reshape(-1, self.envs.num_envs).T
            total_reward_per_env = [
                self.discounted_reward.update(reward_per_step)
                for reward_per_step in total_int_reward.T
            ]
            total_reward_per_env = torch.concatenate(total_reward_per_env)
            mean = total_reward_per_env.mean()
            std = total_reward_per_env.std()
            count = len(total_reward_per_env)

            self.rew_rms.update_from_moments(mean, std**2, count)

        with torch.no_grad():
            mse_norm = mse / self.rew_rms.var.to(x.device).sqrt()

        return mse_norm, mse


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems
