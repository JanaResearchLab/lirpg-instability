import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _make_critic(in_features, *, use_popart, popart_beta):
    if use_popart:
        return PopArt(in_features, 1, beta=popart_beta)
    return layer_init(nn.Linear(in_features, 1), std=1.0)


class PopArt(nn.Module):
    """PopArt value head (van Hasselt et al. 2016); forward returns the de-normalized value."""

    def __init__(self, in_features, out_features=1, beta=3e-4, eps=1e-5, std_init=1.0):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.linear = layer_init(nn.Linear(in_features, out_features), std=std_init)
        self.register_buffer("mu", torch.zeros(out_features))
        self.register_buffer("nu", torch.ones(out_features))
        self.register_buffer("sigma", torch.ones(out_features))

    def forward(self, x):
        return self.sigma * self.linear(x) + self.mu

    def normalize(self, y):
        return (y - self.mu) / self.sigma

    def update_stats(self, targets):
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)

        with torch.no_grad():
            batch_mean = targets.mean(dim=0)
            batch_sq_mean = targets.pow(2).mean(dim=0)

            new_mu = (1.0 - self.beta) * self.mu + self.beta * batch_mean
            new_nu = (1.0 - self.beta) * self.nu + self.beta * batch_sq_mean
            new_sigma = (new_nu - new_mu.pow(2)).clamp_min(self.eps).sqrt()

            weight_ratio = (self.sigma / new_sigma).view(-1, 1)
            bias_scale = self.sigma / new_sigma
            bias_shift = (self.mu - new_mu) / new_sigma

        new_W = weight_ratio * self.linear.weight
        new_b = bias_scale * self.linear.bias + bias_shift

        # Assigned via _parameters so the new non-leaf tensors keep their meta-graph edge.
        self.linear._parameters["weight"] = new_W
        self.linear._parameters["bias"] = new_b

        self.mu = new_mu.detach()
        self.nu = new_nu.detach()
        self.sigma = new_sigma.detach()


class AtariAgent(nn.Module):
    def __init__(
        self,
        envs,
        shared_network=True,
        use_mode_flag=False,
        use_popart=False,
        popart_beta=3e-4,
    ):
        super().__init__()
        self.network_fn = lambda: nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.shared_network = shared_network
        if shared_network:
            self.network = self.actor_network = self.critic_network = self.network_fn()
        else:
            self.actor_network = self.network_fn()
            self.critic_network = self.network_fn()
            self.network = self.critic_network
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = _make_critic(512, use_popart=use_popart, popart_beta=popart_beta)

        self.use_mode_flag = use_mode_flag
        if use_mode_flag:
            self.mode_embedding = nn.Linear(2, 512)

    def _add_mode_embedding(self, hidden, mode_flag):
        if self.use_mode_flag and mode_flag is not None:
            mode_onehot = torch.nn.functional.one_hot(mode_flag.long(), 2).float()
            hidden = hidden + self.mode_embedding(mode_onehot)
        return hidden

    def get_value(self, x, mode_flag=None):
        hidden = self.critic_network(x / 255.0)
        hidden = self._add_mode_embedding(hidden, mode_flag)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None, mode_flag=None):
        x = x / 255.0
        if self.shared_network:
            hidden = self.actor_network(x)
            hidden = self._add_mode_embedding(hidden, mode_flag)
            logits = self.actor(hidden)
            value = self.critic(hidden)
        else:
            actor_hidden = self._add_mode_embedding(self.actor_network(x), mode_flag)
            critic_hidden = self._add_mode_embedding(self.critic_network(x), mode_flag)
            logits = self.actor(actor_hidden)
            value = self.critic(critic_hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value, probs


class MuJoCoAgent(nn.Module):
    def __init__(
        self,
        envs,
        shared_network=False,
        use_popart=False,
        popart_beta=3e-4,
    ):
        super().__init__()
        self.network = self.critic_base = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.critic = _make_critic(64, use_popart=use_popart, popart_beta=popart_beta)

        if shared_network:
            self.network = self.actor_base = self.critic_base
        else:
            self.actor_base = nn.Sequential(
                layer_init(
                    nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
                ),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
            )
        self.actor_mean = layer_init(
            nn.Linear(64, np.prod(envs.single_action_space.shape)),
            std=0.01,
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x, mode_flag=None):
        return self.critic(self.critic_base(x))

    def get_action_and_value(self, x, action=None, mode_flag=None):
        action_mean = self.actor_mean(self.actor_base(x))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.get_value(x),
            probs,
        )


# IMPALA-style ResNet for ProcGen, from CleanRL ppo_procgen.py; see licenses/cleanrl-LICENSE.txt
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(input_shape[0], out_channels, 3, padding=1)
        self.res_block0 = ResidualBlock(out_channels)
        self.res_block1 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class ProcGenAgent(nn.Module):
    def __init__(
        self,
        envs,
        shared_network=True,
        use_mode_flag=False,
        use_popart=False,
        popart_beta=3e-4,
    ):
        super().__init__()
        c, h, w = envs.single_observation_space.shape  # CHW from wrapper
        self.shared_network = shared_network
        self.use_mode_flag = use_mode_flag

        def build_network():
            shape = (c, h, w)
            conv_seqs = []
            for out_channels in [16, 32, 32]:
                conv_seq = ConvSequence(shape, out_channels)
                shape = conv_seq.get_output_shape()
                conv_seqs.append(conv_seq)
            conv_seqs += [
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(shape[0] * shape[1] * shape[2], 256),
                nn.ReLU(),
            ]
            return nn.Sequential(*conv_seqs)

        if shared_network:
            self.network = self.actor_network = self.critic_network = build_network()
        else:
            self.actor_network = build_network()
            self.critic_network = build_network()
            self.network = self.critic_network
        self.actor = layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01)
        self.critic = _make_critic(256, use_popart=use_popart, popart_beta=popart_beta)
        if use_mode_flag:
            self.mode_embedding = nn.Linear(2, 256)

    def _add_mode_embedding(self, hidden, mode_flag):
        if self.use_mode_flag and mode_flag is not None:
            mode_onehot = torch.nn.functional.one_hot(mode_flag.long(), 2).float()
            hidden = hidden + self.mode_embedding(mode_onehot)
        return hidden

    def get_value(self, x, mode_flag=None):
        hidden = self.critic_network(x / 255.0)
        hidden = self._add_mode_embedding(hidden, mode_flag)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None, mode_flag=None):
        x = x / 255.0
        if self.shared_network:
            hidden = self.actor_network(x)
            hidden = self._add_mode_embedding(hidden, mode_flag)
            logits = self.actor(hidden)
            value = self.critic(hidden)
        else:
            actor_hidden = self._add_mode_embedding(self.actor_network(x), mode_flag)
            critic_hidden = self._add_mode_embedding(self.critic_network(x), mode_flag)
            logits = self.actor(actor_hidden)
            value = self.critic(critic_hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value, probs
