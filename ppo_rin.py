# PPO with learned intrinsic rewards, built on CleanRL's PPO (https://github.com/vwxyzjn/cleanrl)
# MIT License, Copyright (c) 2019 CleanRL developers; see licenses/cleanrl-LICENSE.txt.

import copy
import os
import time
from collections import defaultdict

import numpy as np
import psutil
import torch
import torchopt
import tyro
from torch import nn
from torch.distributions import kl_divergence
from torch.utils.tensorboard import SummaryWriter
from torchopt import MetaAdam

import torchopt_patch  # noqa: F401  (adds max_grad_norm to MetaOptimizer.step)
from agents import AtariAgent, MuJoCoAgent, ProcGenAgent
from buffers import MatchReplayBuffer, PPGReplayBuffer
from checkpointing import restore_checkpoint, save_checkpoint
from configs import Args, AtariArgs, MuJoCoArgs, ProcGenArgs
from envs import ProcGenVecEnv, build_vec_env_fn, classify_env, force_real_reset
from intrinsic_reward_module import IntrinsicRewardFn, RNDRewardFn
from utils import set_seed, train_with_logging

process = psutil.Process()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_intrinsic_rewards(
    args,
    r_in_fn,
    *,
    b_inputs,
    b_actions,
    b_mode_flags,
    rewards_ex,
    is_atari,
):
    """r_in(s, a) for a rollout."""
    rewards_in = r_in_fn.get_intrinsic_reward(
        b_inputs, actions=b_actions, mode_flag=b_mode_flags
    )
    if is_atari:
        rewards_in = rewards_in[
            torch.arange(rewards_in.shape[0]),
            b_actions.reshape(-1).long(),
        ]
    rewards_in = rewards_in.reshape(rewards_ex.shape)
    if args.handle_sparse_rex == "r_in_if_r_ex":
        rewards_in = torch.where(
            rewards_ex > 0, rewards_in, torch.zeros_like(rewards_in)
        )
    return rewards_in


def compute_r_mix(
    args, *, use_practice, mode_flags, rewards_ex, rewards_in, r_in_multiplier
):
    """Inner-loop reward lambda_in * r_in + lambda_ex * r_ex, or r_ex / r_in by mode for practice."""
    if use_practice:
        return mode_flags * rewards_ex + (1 - mode_flags) * rewards_in
    return args.r_in_coef * rewards_in * r_in_multiplier + args.r_ex_coef * rewards_ex


def build_gae_coeff_matrix(args, *, values_mix, next_value, dones, next_done, device):
    """Closed-form GAE: A_t = sum_t' coeff_mat[(t,e), t', e] * delta_t', coeff_mat is grad-free."""
    td_mix = torch.zeros_like(values_mix).to(device)
    coeff_mat = torch.zeros((args.batch_size, args.num_steps, args.num_envs)).to(device)
    const = args.gamma * args.gae_lambda

    cm = coeff_mat.view(args.num_steps, args.num_envs, args.num_steps, args.num_envs)
    C = cm.diagonal(dim1=1, dim2=3)  # C[t, t', e] is coeff_mat[t*num_envs + e, t', e]

    last_t = args.num_steps - 1
    nextnonterminal = 1.0 - next_done  # Shape: [1, num_envs]
    nextvalues = next_value
    td_mix[last_t] = args.gamma * nextvalues * nextnonterminal - values_mix[last_t]

    C[last_t, last_t] = 1.0

    for t in reversed(range(args.num_steps - 1)):
        nextnonterminal = 1.0 - dones[t + 1]  # Shape: [num_envs]
        nextvalues = values_mix[t + 1]
        td_mix[t] = args.gamma * nextvalues * nextnonterminal - values_mix[t]

        C[t, t] = 1.0
        C[t, t + 1 :] = (const * nextnonterminal) * C[t + 1, t + 1 :]

    return td_mix, coeff_mat


def extrinsic_value(agent, r_in_fn, obs, *, mode_flag, args):
    """v_ex from the agent's match-mode value head (practice_use_agent_vex) or r_in_fn's v_ex head."""
    use_practice = args.practice_steps > 0
    if use_practice and args.practice_use_agent_vex:
        match_mf = torch.ones_like(mode_flag)
        return agent.get_value(obs, mode_flag=match_mf)
    return r_in_fn.get_extrinsic_value(obs, mode_flag=mode_flag)


def stop_all_gradients(agent, optim, r_in_fn, r_in_optimizer):
    """Detach the meta-graph on the agent/optimizer/r_in_fn between meta-steps."""
    torchopt.stop_gradient(agent)
    torchopt.stop_gradient(optim)
    torchopt.stop_gradient(r_in_fn)
    torchopt.stop_gradient(r_in_optimizer)


def _actor_head_params(agent):
    """Policy head parameters (actor, or actor_mean plus actor_logstd on MuJoCo)."""
    if hasattr(agent, "actor"):
        return list(agent.actor.parameters())
    params = list(agent.actor_mean.parameters())
    if hasattr(agent, "actor_logstd"):
        params.append(agent.actor_logstd)
    return params


def _agent_policy_value_encoders(agent):
    """(policy, value) encoders, the same object under a shared encoder."""
    value_enc = agent.network
    for attr in ("actor_network", "actor_base"):
        enc = getattr(agent, attr, None)
        if enc is not None:
            return enc, value_enc
    return value_enc, value_enc


def pcgrad_reweight(l_pol, l_val, params):
    """First-order PCGrad (Yu et al. 2020) as detached weights on the policy and value losses."""

    def _flat_grad(scalar):
        grads = torch.autograd.grad(
            scalar, params, retain_graph=True, allow_unused=True
        )
        return torch.nn.utils.parameters_to_vector(
            [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]
        )

    g_pol = _flat_grad(l_pol)
    g_val = _flat_grad(l_val)
    dot = torch.dot(g_pol, g_val)
    sq_pol = torch.dot(g_pol, g_pol)
    sq_val = torch.dot(g_val, g_val)
    cos = (dot / (sq_pol.sqrt() * sq_val.sqrt() + 1e-12)).item()
    conflict = dot.item() < 0.0
    if conflict:
        # c_pol multiplies g_val in g_pol', c_val multiplies g_pol in g_val'
        c_pol = (dot / (sq_val + 1e-12)).item()
        c_val = (dot / (sq_pol + 1e-12)).item()
        w_pol = 1.0 - c_val
        w_val = 1.0 - c_pol
    else:
        w_pol = 1.0
        w_val = 1.0
    stats = {
        "pcgrad_cosine": cos,
        "pcgrad_conflict": 1.0 if conflict else 0.0,
        "pcgrad_w_pol": w_pol,
        "pcgrad_w_val": w_val,
    }
    return w_pol, w_val, stats


class GradNorm:
    """GradNorm (Chen et al. 2018) loss weights for the policy/value split at a shared encoder."""

    def __init__(self, network, *, alpha, lr, device):
        # MetaAdam replaces the parameters every inner step, so they are re-read on each reweight
        self.network = network
        self.alpha = alpha
        self.w = torch.ones(2, device=device, requires_grad=True)  # [w_pol, w_val]
        self.optimizer = torch.optim.Adam([self.w], lr=lr)
        self.l0 = None  # per-task losses L_i(0), set on first reweight()

    def state_dict(self):
        return {
            "w": self.w.detach().clone(),
            "optimizer": self.optimizer.state_dict(),
            "l0": self.l0,
        }

    def load_state_dict(self, sd):
        with torch.no_grad():
            self.w.copy_(sd["w"].to(self.w.device))
        self.optimizer.load_state_dict(sd["optimizer"])
        self.l0 = sd["l0"]

    def _grad_norm(self, loss):
        params = list(self.network.parameters())
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        vec = torch.nn.utils.parameters_to_vector(
            [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]
        )
        return vec.norm()

    def reweight(self, l_pol, l_val):
        n_pol = self._grad_norm(l_pol).detach()
        n_val = self._grad_norm(l_val).detach()

        G_pol = self.w[0] * n_pol
        G_val = self.w[1] * n_val
        G_bar = (0.5 * (G_pol + G_val)).detach()

        l_pol_val = l_pol.item()
        l_val_val = l_val.item()
        if self.l0 is None:
            self.l0 = (abs(l_pol_val) + 1e-8, abs(l_val_val) + 1e-8)
        if self.alpha != 0.0:
            lt_pol = abs(l_pol_val) / self.l0[0]
            lt_val = abs(l_val_val) / self.l0[1]
            lt_mean = 0.5 * (lt_pol + lt_val) + 1e-8
            r_pol = (lt_pol / lt_mean) ** self.alpha
            r_val = (lt_val / lt_mean) ** self.alpha
        else:
            r_pol = r_val = 1.0
        target_pol = (G_bar * r_pol).detach()
        target_val = (G_bar * r_val).detach()

        l_grad = (G_pol - target_pol).abs() + (G_val - target_val).abs()
        self.optimizer.zero_grad()
        l_grad.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.w.clamp_(min=1e-8)
            self.w.mul_(2.0 / self.w.sum())

        w_pol = self.w[0].item()
        w_val = self.w[1].item()
        stats = {
            "gradnorm_w_pol": w_pol,
            "gradnorm_w_val": w_val,
            "gradnorm_grad_norm_pol": n_pol.item(),
            "gradnorm_grad_norm_val": n_val.item(),
            "gradnorm_target_pol": target_pol.item(),
            "gradnorm_target_val": target_val.item(),
            "gradnorm_loss": l_grad.item(),
            "gradnorm_ratio_pol": r_pol if self.alpha != 0.0 else 1.0,
            "gradnorm_ratio_val": r_val if self.alpha != 0.0 else 1.0,
        }
        return w_pol, w_val, stats


# Set by __main__ when resuming, consumed by train()
_LOADED_CHECKPOINT = None


def _encoder_grad_vec(scalar_loss, params):
    """Flat gradient of scalar_loss w.r.t. params, with zeros for unused params."""
    grads = torch.autograd.grad(
        scalar_loss, params, retain_graph=True, allow_unused=True
    )
    return torch.nn.utils.parameters_to_vector(
        [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]
    )


def _cos(a, b):
    """Cosine similarity between two flat gradient vectors, as a Python float."""
    return torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def train(args: Args, writer: SummaryWriter | None = None):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    env_kinds = classify_env(args.env_id)
    is_atari = env_kinds["is_atari"]
    is_procgen = env_kinds["is_procgen"]

    vec_env_fn = build_vec_env_fn(args, run_name)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    global _LOADED_CHECKPOINT
    ckpt = _LOADED_CHECKPOINT
    resuming = ckpt is not None

    obs_mean = obs_std = np.zeros(1)
    if resuming:
        obs_mean = ckpt["obs_mean"]
        obs_std = ckpt["obs_std"]
    elif args.r_rnd_coef > 0:
        set_seed(args.seed)
        envs = vec_env_fn(n_envs=1)
        obs_shape = envs.single_observation_space.shape
        obs, _ = envs.reset(seed=args.seed)

        obs_mean = np.zeros(obs_shape, dtype=np.float64)
        obs_var = np.zeros(obs_shape, dtype=np.float64)
        count = 0

        print("Collecting observations for RND normalization...")
        for _ in range(args.rnd_warmup_steps):
            count += 1
            action = [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            obs, _, _, _, _ = envs.step(action)
            obs = obs.squeeze(0)

            # Welford's algorithm
            delta = obs - obs_mean
            obs_mean += delta / count
            delta2 = obs - obs_mean
            obs_var += delta * delta2

        obs_std = np.sqrt(obs_var / count)

    set_seed(args.seed)
    envs = vec_env_fn()

    print("Starting", flush=True)

    shared_networks = args.shared_networks.split(",")
    shared_pi_vmix = "pi-vmix" in shared_networks or args.use_ppg
    use_practice = args.practice_steps > 0

    if args.pcgrad and not shared_pi_vmix:
        print(
            "WARNING: --pcgrad has no effect without a shared policy/value "
            "encoder (add 'pi-vmix' to --shared_networks); running as a plain "
            "baseline.",
            flush=True,
        )

    if args.gradnorm and not shared_pi_vmix:
        print(
            "WARNING: --gradnorm has no effect without a shared policy/value "
            "encoder (add 'pi-vmix' to --shared_networks); running as a plain "
            "baseline.",
            flush=True,
        )
    assert not (args.pcgrad and args.use_ppg), (
        "pcgrad is mutually exclusive with use_ppg (they are alternative "
        "policy/value decoupling schemes)"
    )
    assert not (args.gradnorm and (args.pcgrad or args.use_ppg)), (
        "gradnorm is mutually exclusive with pcgrad and use_ppg (they are "
        "alternative policy/value balancing / decoupling schemes)"
    )
    assert not (args.use_ppg and args.use_target_rin), (
        "use_target_rin is only supported without use_ppg"
    )
    assert "rin-pi" not in shared_networks, "'rin-pi' sharing is not supported"
    assert not (use_practice and args.freeze_rin_frac < 1.0), (
        "--freeze_rin_frac is not supported with practice-match"
    )
    if is_atari:
        agent = AtariAgent(
            envs,
            shared_network=shared_pi_vmix,
            use_mode_flag=use_practice,
            use_popart=args.use_popart,
            popart_beta=args.popart_beta,
        ).to(device)
    elif is_procgen:
        agent = ProcGenAgent(
            envs,
            shared_network=shared_pi_vmix,
            use_mode_flag=use_practice,
            use_popart=args.use_popart,
            popart_beta=args.popart_beta,
        ).to(device)
    else:
        agent = MuJoCoAgent(
            envs,
            shared_network=shared_pi_vmix,
            use_popart=args.use_popart,
            popart_beta=args.popart_beta,
        ).to(device)

    optim = MetaAdam(agent, lr=args.learning_rate, eps=1e-5, use_accelerated_op=True)

    gradnorm = (
        GradNorm(
            agent.network,
            alpha=args.gradnorm_alpha,
            lr=args.gradnorm_lr,
            device=device,
        )
        if args.gradnorm
        else None
    )

    print("Agent Initialized", flush=True)

    if args.agent_base_model_path:
        base_encoder_sd = torch.load(args.agent_base_model_path, map_location=device)
        # Under a shared encoder policy_encoders is empty
        policy_enc, value_enc = _agent_policy_value_encoders(agent)
        value_encoders = [value_enc]
        policy_encoders = [] if policy_enc is value_enc else [policy_enc]
        if args.agent_base_model_target == "policy":
            assert policy_encoders, (
                "--agent_base_model_target policy requires a policy encoder that is "
                "separate from the value encoder; drop 'pi-vmix' from "
                "--shared_networks. With a shared encoder there is one module, so "
                "the value encoder would be pretrained too."
            )
            encoders = policy_encoders
        else:
            encoders = value_encoders + policy_encoders
        for enc in encoders:
            enc.load_state_dict(base_encoder_sd)
        print(
            f"Loaded agent base model from path: {args.agent_base_model_path} "
            f"into {len(encoders)} encoder(s) "
            f"(target={args.agent_base_model_target})",
            flush=True,
        )
    r_in_fn = IntrinsicRewardFn(
        envs,
        r_in_per_action=is_atari,
        shared_rin_vex="rin-vex" in shared_networks,
        activation_fn=args.r_in_activation,
        use_mode_flag=use_practice,
    ).to(device)

    def r_in_optim_fn(r_in_fn, lr=args.r_in_learning_rate):
        # With practice_use_agent_vex the agent's value head is V_ex
        include_v_ex = not (use_practice and args.practice_use_agent_vex)
        params = r_in_fn.get_meta_params(include_v_ex=include_v_ex)
        return torchopt.Adam(params, lr=lr, eps=1e-5)

    r_in_optimizer = r_in_optim_fn(r_in_fn)
    print("Intrinsic Reward Function Initialized", flush=True)

    # DQN-style slow copy of r_in_fn that supplies the value-loss returns
    r_in_target_fn = None
    if args.use_target_rin:
        r_in_target_fn = copy.deepcopy(r_in_fn)
        for param in r_in_target_fn.parameters():
            param.requires_grad = False
        print("Target Intrinsic Reward Function Initialized", flush=True)

    r_rnd_fn = RNDRewardFn(envs, obs_mean, obs_std, args.gamma).to(device)
    r_rnd_optimizer = torch.optim.Adam(
        r_rnd_fn.parameters(), lr=args.r_rnd_learning_rate
    )
    print("Hand-designed Reward Function Initialized", flush=True)

    set_seed(args.seed)

    # ALGO Logic: Storage setup
    batch_shape = (args.num_steps, args.num_envs)
    obs_shape = envs.single_observation_space.shape
    obs = torch.zeros(batch_shape + obs_shape).to(device)
    actions = torch.zeros(batch_shape + envs.single_action_space.shape).to(device)
    next_obss = torch.zeros_like(obs).to(device)
    logprobs = torch.zeros(batch_shape).to(device)
    rewards_ex = torch.zeros(batch_shape).to(device)
    dones = torch.zeros(batch_shape).to(device)
    # Real terminations only (excludes truncation)
    terms_ex = torch.zeros(batch_shape).to(device)
    terms_ex_np = np.zeros(batch_shape, dtype=np.float32)
    values_mix = torch.zeros(batch_shape).to(device)
    values_ex = torch.zeros(batch_shape).to(device)
    td_mix = torch.zeros(batch_shape).to(device)

    # Used for correcting for truncated episodes
    # https://github.com/DLR-RM/stable-baselines3/blob/9e5ccc1832366d55abbf31b561ea9dcfbfdfa485/stable_baselines3/common/on_policy_algorithm.py#L245
    next_trunc_values_mix = torch.zeros(batch_shape).to(device)
    next_trunc_values_ex = torch.zeros(batch_shape).to(device)

    if use_practice:
        match_tags = torch.ones(args.num_envs, device=device)  # 1=match, 0=practice
        practice_step_counts = np.zeros(args.num_envs, dtype=np.int64)
        mode_flags = torch.zeros(batch_shape, device=device)  # (num_steps, num_envs)
        match_replay_buffer = MatchReplayBuffer(
            capacity=args.practice_buffer_size,
            obs_shape=envs.single_observation_space.shape,
            action_shape=envs.single_action_space.shape,
        )
        print(
            f"Practice-match enabled: practice_steps={args.practice_steps}, "
            f"buffer_size={args.practice_buffer_size}, batch_size={args.practice_batch_size}",
            flush=True,
        )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    # Env steps in match mode (equals global_step without practice)
    match_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # For reward frequency
    delay_r_ex = np.zeros([args.num_envs])
    delay_step = np.zeros([args.num_envs])

    return_timesteps = []

    # DCPG value-phase buffer: the last t_off rollouts.
    assert not args.use_ppg or envs.single_observation_space.dtype == np.uint8, (
        "the DCPG buffer stores uint8 observations (Atari, ProcGen)"
    )
    rb_ppg = (
        PPGReplayBuffer(
            capacity=args.t_off * args.num_envs * args.num_steps,
            obs_shape=envs.single_observation_space.shape,
            batch_size=args.m_v,
            track_mode_flags=use_practice,
        )
        if args.use_ppg
        else None
    )

    start_iteration = 1
    _saved_adam_snr = None
    if resuming:
        # Restore from the checkpoint, the loop resumes at ckpt["iteration"] + 1
        restored = restore_checkpoint(
            ckpt,
            args=args,
            device=device,
            agent=agent,
            r_in_fn=r_in_fn,
            r_in_target_fn=r_in_target_fn,
            r_rnd_fn=r_rnd_fn,
            optim=optim,
            r_in_optimizer=r_in_optimizer,
            r_rnd_optimizer=r_rnd_optimizer,
            rb_ppg=rb_ppg,
            envs=envs,
        )
        rb_ppg = restored["rb_ppg"]
        envs = restored["envs"]
        next_obs = restored["next_obs"]
        next_done = restored["next_done"]
        global_step = restored["global_step"]
        match_step = restored["match_step"]
        return_timesteps = restored["return_timesteps"]
        delay_r_ex = restored["delay_r_ex"]
        delay_step = restored["delay_step"]
        start_iteration = restored["start_iteration"]
        start_time = restored["start_time"]
        _saved_adam_snr = restored["adam_snr_state"]
        if gradnorm is not None and restored["gradnorm_state"] is not None:
            gradnorm.load_state_dict(restored["gradnorm_state"])
        if use_practice and restored["practice_state"] is not None:
            ps = restored["practice_state"]
            match_tags = ps["match_tags"].to(device)
            practice_step_counts = ps["practice_step_counts"]
            match_replay_buffer = ps["match_replay_buffer"]
        _LOADED_CHECKPOINT = None
        del ckpt

    # Adam-style EMAs of policy/value gradients per param group, for SNR logging
    adam_snr_state = {
        "step": 0,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "m": {},
        "v": {},
    }
    if _saved_adam_snr is not None:
        adam_snr_state.update(_saved_adam_snr)

    def _update_adam_snr(name, grad_vec):
        st = adam_snr_state
        if name not in st["m"]:
            st["m"][name] = torch.zeros_like(grad_vec)
            st["v"][name] = torch.zeros_like(grad_vec)
        st["m"][name].mul_(st["beta1"]).add_(grad_vec, alpha=1 - st["beta1"])
        st["v"][name].mul_(st["beta2"]).addcmul_(
            grad_vec, grad_vec, value=1 - st["beta2"]
        )
        t = st["step"]
        m_hat = st["m"][name] / (1 - st["beta1"] ** t)
        v_hat = st["v"][name] / (1 - st["beta2"] ** t)
        return (m_hat.abs() / (v_hat.sqrt() + st["eps"])).mean().item()

    def log_episode(ep_r, ep_l):
        print(
            f"global_step={global_step}, match_step={match_step}, episodic_return={ep_r}, episodic_length={ep_l}",
            flush=True,
        )
        if writer is not None:
            writer.add_scalar("charts/episodic_return", ep_r, global_step)
            writer.add_scalar("charts/episodic_length", ep_l, global_step)
            writer.add_scalar("charts/episodic_return_per_match_step", ep_r, match_step)
            writer.add_scalar("charts/episodic_length_per_match_step", ep_l, match_step)
        return_timesteps.append((ep_r.item(), global_step))

    def log_stats(name, tensor, stats=("mean", "max", "min", "std", "median")):
        for stat in stats:
            writer.add_scalar(
                f"charts/{stat}_{name}", getattr(tensor, stat)().item(), global_step
            )

    for iteration in range(start_iteration, args.num_iterations + 1):
        if (
            args.max_match_timesteps is not None
            and match_step >= args.max_match_timesteps
        ):
            print(
                f"Reached max_match_timesteps={args.max_match_timesteps} "
                f"(match_step={match_step}); stopping training.",
                flush=True,
            )
            break

        n_truncated = 0
        n_terminated = 0

        if args.freeze_rin_frac < 1.0:
            freeze_iteration = int(args.freeze_rin_frac * args.num_iterations)
            if iteration < freeze_iteration:
                for param in r_in_fn.parameters():
                    param.requires_grad = True
            else:
                args.freeze_rin_frac = 1.0
                print("Freezing r_in function", flush=True)
                args.meta_update = False

        # Annealing the rate if instructed to do so.
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        lrnow = args.learning_rate
        if args.anneal_lr and iteration > 1:
            lrnow = frac * args.learning_rate
            optim_state = optim.state_dict()
            optim = MetaAdam(agent, lr=lrnow, eps=1e-5, use_accelerated_op=True)
            optim.load_state_dict(optim_state)

            r_in_optim_state = r_in_optimizer.state_dict()
            r_in_optimizer = r_in_optim_fn(r_in_fn, lr=frac * args.r_in_learning_rate)
            r_in_optimizer.load_state_dict(r_in_optim_state)
        clipnow = args.clip_coef

        for step in range(args.num_steps):
            global_step += args.num_envs
            if use_practice:
                match_step += int(match_tags.sum().item())
            else:
                match_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            if use_practice:
                mode_flags[step] = match_tags.clone()

            # ALGO LOGIC: action logic
            with torch.no_grad():
                mf = match_tags if use_practice else None
                action, logprob, _, value, _ = agent.get_action_and_value(
                    next_obs, mode_flag=mf
                )
                values_mix[step] = value.flatten()
                values_ex[step] = extrinsic_value(
                    agent, r_in_fn, next_obs, mode_flag=mf, args=args
                ).flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )

            n_truncated += np.sum(truncations)
            n_terminated += np.sum(terminations)

            terms_ex_np[step] = terminations
            next_done_np = np.logical_or(terminations, truncations)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done_np).to(device),
            )

            # RecordEpisodeStatistics wraps below EpisodicLifeEnv, so episode events are real game-overs
            real_done_np = np.zeros(args.num_envs, dtype=bool)
            if "final_info" in infos and isinstance(infos["final_info"], dict):
                # Vector-style infos (new gymnasium)
                if "_episode" in infos["final_info"]:
                    real_done_np = np.asarray(
                        infos["final_info"]["_episode"], dtype=bool
                    )
            elif "final_info" in infos:
                # List-style infos (older gymnasium)
                for env_idx, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        real_done_np[env_idx] = True

            if args.handle_trunc:
                # Note: ignoring the next state for terminated episodes because value is set to zero anyway
                next_obss[step] = next_obs
                next_trunc_values_mix[step] = 0
                next_trunc_values_ex[step] = 0
                for n in np.nonzero(truncations)[0]:
                    if "final_observation" in infos:
                        final_obs = infos["final_observation"][n]
                    elif "final_obs" in infos:
                        final_obs = infos["final_obs"][n]
                    else:
                        final_obs = next_obs[n].cpu().numpy()
                    next_obss[step, n] = torch.Tensor(np.array(final_obs)).to(device)
                    mf_n = match_tags[n : n + 1] if use_practice else None
                    with torch.no_grad():
                        next_trunc_values_mix[step, n] = agent.get_value(
                            next_obss[step, n].unsqueeze(0), mode_flag=mf_n
                        ).item()
                        next_trunc_values_ex[step, n] = extrinsic_value(
                            agent,
                            r_in_fn,
                            next_obss[step, n].unsqueeze(0),
                            mode_flag=mf_n,
                            args=args,
                        ).item()

            if use_practice:
                next_done_np = np.logical_or(terminations, truncations)
                for n in range(args.num_envs):
                    if match_tags[n].item() == 0:
                        practice_step_counts[n] += 1
                        if practice_step_counts[n] >= args.practice_steps:
                            # End of practice: reset the env so the match starts a fresh episode
                            if not next_done_np[n]:
                                # Bootstrap the practice end-state for GAE truncation
                                mf_practice = match_tags[n : n + 1]
                                with torch.no_grad():
                                    next_trunc_values_mix[step, n] = agent.get_value(
                                        next_obs[n].unsqueeze(0),
                                        mode_flag=mf_practice,
                                    ).item()
                                    next_trunc_values_ex[step, n] = extrinsic_value(
                                        agent,
                                        r_in_fn,
                                        next_obs[n].unsqueeze(0),
                                        mode_flag=mf_practice,
                                        args=args,
                                    ).item()
                                # ProcGenVecEnv cannot reset one sub-env, so use the cached observation instead
                                if isinstance(envs, ProcGenVecEnv):
                                    if envs._last_obs_processed is not None:
                                        reset_obs = envs._last_obs_processed[n]
                                    else:
                                        reset_obs = next_obs[n].cpu().numpy()
                                    next_obs[n] = torch.Tensor(np.array(reset_obs)).to(
                                        device
                                    )
                                    next_done[n] = 1.0
                                else:
                                    reset_obs, _ = force_real_reset(envs.envs[n])
                                    next_obs[n] = torch.Tensor(np.array(reset_obs)).to(
                                        device
                                    )
                                    next_done[n] = 1.0
                            match_tags[n] = 1
                    elif real_done_np[n]:
                        match_tags[n] = 0
                        practice_step_counts[n] = 0

            delay_r_ex += reward
            delay_step += 1
            done_np = (
                next_done.detach().cpu().numpy() > 0 if use_practice else next_done_np
            )
            emit = done_np | (delay_step == args.reward_freq)
            rewards_ex[step] = torch.as_tensor(
                np.where(emit, delay_r_ex, 0.0), dtype=rewards_ex.dtype, device=device
            )
            delay_r_ex[emit] = 0
            delay_step[emit] = 0

            # Skip episodes that ended in practice mode
            if "final_info" in infos and "episode" in infos["final_info"]:
                ep_info = infos["final_info"]["episode"]
                for env_idx, ep_done in enumerate(infos["final_info"]["_episode"]):
                    if not ep_done:
                        continue
                    if use_practice and mode_flags[step, env_idx].item() == 0:
                        continue
                    log_episode(ep_info["r"][env_idx], ep_info["l"][env_idx])
            elif "final_info" in infos:
                for env_idx, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        if use_practice and mode_flags[step, env_idx].item() == 0:
                            continue
                        log_episode(info["episode"]["r"], info["episode"]["l"])

        if writer is not None:
            writer.add_scalar("charts/n_terminated", n_terminated, global_step)
            writer.add_scalar("charts/n_truncated", n_truncated, global_step)
            writer.add_scalar(
                "charts/pct_truncated",
                n_truncated / (n_terminated + n_truncated + 1e-8),
                global_step,
            )

        if args.random_policy:
            sps = int(global_step / (time.time() - start_time))
            print("SPS:", sps, flush=True)
            if writer:
                writer.add_scalar("charts/SPS", sps, global_step)
                log_stats("r_ex", rewards_ex, stats=("mean", "max", "min"))
                writer.add_scalar(
                    "charts/memory_usage",
                    process.memory_info().rss / 1024**2,
                    global_step,
                )
            continue

        # bootstrap value if not done
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_next_obs = next_obss.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        # torch.tensor copies, as_tensor would alias a numpy buffer the PPG replay buffer keeps
        terms_ex = torch.tensor(terms_ex_np, device=device)
        b_terms_ex = terms_ex.reshape(-1)
        b_mode_flags = mode_flags.reshape(-1) if use_practice else None

        # Atari r_in has a per-action head, MuJoCo r_in reads state and action concatenated
        if (
            is_atari
            or "rin-vex" in args.shared_networks
            or len(envs.single_observation_space.shape) > 1
        ):
            b_inputs = b_obs
        else:
            b_inputs = torch.concat([b_obs, b_actions], dim=1)

        r_in_multiplier = 1.0

        with torch.no_grad():
            mf = match_tags if use_practice else None
            next_value = agent.get_value(next_obs, mode_flag=mf).reshape(1, -1)

            td_mix, coeff_mat = build_gae_coeff_matrix(
                args,
                values_mix=values_mix,
                next_value=next_value,
                dones=dones,
                next_done=next_done,
                device=device,
            )

            if args.r_in_coef or use_practice:
                rewards_in = compute_intrinsic_rewards(
                    args,
                    r_in_fn,
                    b_inputs=b_inputs,
                    b_actions=b_actions,
                    b_mode_flags=b_mode_flags,
                    rewards_ex=rewards_ex,
                    is_atari=is_atari,
                )
            else:
                rewards_in = torch.zeros_like(rewards_ex).to(device)
            if args.r_rnd_coef:
                rewards_rnd, mse_rnd = r_rnd_fn.get_intrinsic_reward(b_obs)
                rewards_rnd = rewards_rnd.reshape(rewards_ex.shape)
            else:
                mse_rnd = rewards_rnd = torch.zeros_like(rewards_ex).to(device)
            if args.handle_sparse_rex == "scale_r_in_by_density":
                # r_in is scaled by the fraction of non-zero r_ex in the batch
                r_in_multiplier = (rewards_ex != 0).float().mean().item()
            r_mix = compute_r_mix(
                args,
                use_practice=use_practice,
                mode_flags=mode_flags if use_practice else None,
                rewards_ex=rewards_ex,
                rewards_in=rewards_in,
                r_in_multiplier=r_in_multiplier,
            )

            delta = (
                r_mix
                + args.gamma * next_trunc_values_mix
                + args.r_rnd_coef * rewards_rnd
                + td_mix
            )
            delta_reshaped = delta.unsqueeze(0)
            advantages_mix = (
                (coeff_mat * delta_reshaped)
                .sum(dim=(1, 2))
                .reshape((args.num_steps, args.num_envs))
            )
            returns_mix = advantages_mix + values_mix

        # flatten the batch
        b_logprobs = logprobs.reshape(-1)
        b_values_mix = values_mix.reshape(-1)
        b_returns_mix = returns_mix.reshape(-1)

        if args.use_ppg:
            rb_ppg.push(obs=b_obs, returns=b_returns_mix, mode_flags=b_mode_flags)

        if args.meta_update or use_practice:
            with torch.no_grad():
                mf = match_tags if use_practice else None
                next_value_ex = extrinsic_value(
                    agent, r_in_fn, next_obs, mode_flag=mf, args=args
                ).reshape(1, -1)
                advantages_ex = torch.zeros_like(rewards_ex).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value_ex
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values_ex[t + 1]
                    delta = (
                        rewards_ex[t]
                        + args.gamma_outer * next_trunc_values_ex[t]
                        + args.gamma_outer * nextvalues * nextnonterminal
                        - values_ex[t]
                    )
                    advantages_ex[t] = lastgaelam = (
                        delta
                        + args.gamma_outer
                        * args.gae_lambda_outer
                        * nextnonterminal
                        * lastgaelam
                    )
                returns_ex = advantages_ex + values_ex

            # flatten the batch
            b_advantages_ex = advantages_ex.reshape(-1)
            b_returns_ex = returns_ex.reshape(-1)
            b_values_ex = values_ex.reshape(-1)

            # Store raw match transitions, returns are recomputed at meta-update time
            if use_practice:
                match_mask = b_mode_flags == 1
                if match_mask.any():
                    match_replay_buffer.push(
                        b_obs[match_mask],
                        b_next_obs[match_mask],
                        b_actions[match_mask],
                        b_logprobs[match_mask],
                        rewards_ex.reshape(-1)[match_mask],
                        b_terms_ex[match_mask],
                    )

        if args.r_rnd_coef:
            for _ in range(args.rnd_update_epochs):
                r_rnd_optimizer.zero_grad()
                loss = r_rnd_fn.get_intrinsic_reward(b_obs, update_rms=False)[1].mean()
                loss.backward()
                r_rnd_optimizer.step()

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        inner_losses = defaultdict(list)
        outer_losses = defaultdict(list)
        # Pre-update snapshot for the drift logs
        _pre_update_agent = copy.deepcopy(agent) if args.log_grad_stats else None
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_mf = b_mode_flags[mb_inds] if use_practice else None
                _, newlogprob, entropy, newvalue, new_dist = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds], mode_flag=mb_mf
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    inner_old_approx_kl = (-logratio).mean()
                    inner_approx_kl = ((ratio - 1) - logratio).mean()
                    inner_clipfrac = (
                        ((ratio - 1.0).abs() > clipnow).float().mean().item()
                    )

                    inner_losses["approx_kl"].append(inner_approx_kl.item())
                    inner_losses["old_approx_kl"].append(inner_old_approx_kl.item())
                    inner_losses["clipfrac"].append(inner_clipfrac)

                # Recompute r_in and GAE returns in the meta-graph so the inner update depends on r_in_fn
                if args.r_in_coef or use_practice:
                    rewards_in = compute_intrinsic_rewards(
                        args,
                        r_in_fn,
                        b_inputs=b_inputs,
                        b_actions=b_actions,
                        b_mode_flags=b_mode_flags,
                        rewards_ex=rewards_ex,
                        is_atari=is_atari,
                    )
                else:
                    rewards_in = torch.zeros_like(rewards_ex).to(device)
                r_mix = compute_r_mix(
                    args,
                    use_practice=use_practice,
                    mode_flags=mode_flags if use_practice else None,
                    rewards_ex=rewards_ex,
                    rewards_in=rewards_in,
                    r_in_multiplier=r_in_multiplier,
                )
                delta = (
                    r_mix
                    + args.gamma * next_trunc_values_mix
                    + args.r_rnd_coef * rewards_rnd.detach()
                    + td_mix
                )
                advantages_mix = (coeff_mat[mb_inds] * delta.unsqueeze(0)).sum(
                    dim=(1, 2)
                )
                returns_mix = advantages_mix + b_values_mix[mb_inds]

                mb_returns_mix = returns_mix.reshape(-1)
                mb_advantages = advantages_mix.reshape(-1)

                # Value returns from the target r_in network
                mb_returns_mix_target = None
                if args.use_target_rin:
                    with torch.no_grad():
                        if not (args.r_in_coef or use_practice):
                            rewards_in_target = rewards_in
                        else:
                            rewards_in_target = compute_intrinsic_rewards(
                                args,
                                r_in_target_fn,
                                b_inputs=b_inputs,
                                b_actions=b_actions,
                                b_mode_flags=b_mode_flags,
                                rewards_ex=rewards_ex,
                                is_atari=is_atari,
                            )
                        r_mix_target = compute_r_mix(
                            args,
                            use_practice=use_practice,
                            mode_flags=mode_flags if use_practice else None,
                            rewards_ex=rewards_ex,
                            rewards_in=rewards_in_target,
                            r_in_multiplier=r_in_multiplier,
                        )
                        delta_target = (
                            r_mix_target
                            + args.gamma * next_trunc_values_mix
                            + args.r_rnd_coef * rewards_rnd
                            + td_mix
                        )
                        advantages_mix_target = (
                            coeff_mat[mb_inds] * delta_target.unsqueeze(0)
                        ).sum(dim=(1, 2))
                        mb_returns_mix_target = (
                            advantages_mix_target + b_values_mix[mb_inds]
                        ).reshape(-1)

                if args.norm_adv:
                    if use_practice:
                        # Normalize advantages within each mode
                        match_mask_norm = mb_mf == 1
                        practice_mask_norm = mb_mf == 0
                        normalized = mb_advantages.clone()
                        if match_mask_norm.any():
                            m_adv = mb_advantages[match_mask_norm]
                            if m_adv.numel() > 1:
                                normalized[match_mask_norm] = (m_adv - m_adv.mean()) / (
                                    m_adv.std() + 1e-8
                                )
                            else:
                                normalized[match_mask_norm] = m_adv - m_adv.mean()
                        if practice_mask_norm.any():
                            p_adv = mb_advantages[practice_mask_norm]
                            if p_adv.numel() > 1:
                                normalized[practice_mask_norm] = (
                                    p_adv - p_adv.mean()
                                ) / (p_adv.std() + 1e-8)
                            else:
                                normalized[practice_mask_norm] = p_adv - p_adv.mean()
                        mb_advantages = normalized
                    else:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clipnow, 1 + clipnow)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (in normalized space under PopArt)
                newvalue = newvalue.view(-1)
                # use_target_rin takes value targets from the grad-free target r_in network
                if args.use_target_rin:
                    value_target = mb_returns_mix_target
                    inner_losses["value_target_gap"].append(
                        (mb_returns_mix.detach() - mb_returns_mix_target)
                        .abs()
                        .mean()
                        .item()
                    )
                else:
                    value_target = mb_returns_mix
                if args.use_popart:
                    pred_for_loss = agent.critic.normalize(newvalue)
                    target_for_loss = agent.critic.normalize(value_target)
                    rollout_for_loss = agent.critic.normalize(b_values_mix[mb_inds])
                else:
                    pred_for_loss = newvalue
                    target_for_loss = value_target
                    rollout_for_loss = b_values_mix[mb_inds]

                if args.use_ppg:
                    # DCPG policy phase: value regularizer toward the rollout values
                    v_loss = (
                        0.5
                        * ((pred_for_loss - rollout_for_loss) ** 2).mean()
                        / args.vf_coef
                        * args.beta_v
                    )
                elif args.clip_vloss:
                    v_loss_unclipped = (pred_for_loss - target_for_loss) ** 2
                    v_clipped = rollout_for_loss + torch.clamp(
                        pred_for_loss - rollout_for_loss,
                        -clipnow,
                        clipnow,
                    )
                    v_loss_clipped = (v_clipped - target_for_loss) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((pred_for_loss - target_for_loss) ** 2).mean()

                if use_practice:
                    # Entropy bonus on practice steps only, as in Rajendran et al. (2020)
                    practice_mask = 1.0 - mb_mf
                    practice_count = practice_mask.sum()
                    entropy_loss = (practice_mask * entropy).sum() / (
                        practice_count + 1e-8
                    )
                else:
                    entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                if args.use_ppg and args.log_grad_stats:
                    # DCPG gradient decomposition at the shared encoder, policy phase
                    _enc = list(agent.network.parameters())
                    _g_V_loss = 0.5 * ((newvalue - mb_returns_mix.detach()) ** 2).mean()
                    _g_vr_loss = (
                        0.5 * ((newvalue - b_values_mix[mb_inds].detach()) ** 2).mean()
                    )
                    with torch.no_grad():
                        _g_pi = _encoder_grad_vec(pg_loss, _enc)
                        _g_V = _encoder_grad_vec(_g_V_loss, _enc)
                        _g_vr = _encoder_grad_vec(_g_vr_loss, _enc)
                        _g = _encoder_grad_vec(loss, _enc)
                        for _k, (_a, _b) in {
                            "cos_g_gpi": (_g, _g_pi),
                            "cos_g_gV": (_g, _g_V),
                            "cos_g_gvr": (_g, _g_vr),
                            "cos_gpi_gvr": (_g_pi, _g_vr),
                            "cos_gpi_gV": (_g_pi, _g_V),
                            "cos_gV_gvr": (_g_V, _g_vr),
                        }.items():
                            inner_losses[f"phasic_pol/{_k}"].append(_cos(_a, _b))
                if args.log_grad_stats:
                    # Under phasic optimization the value gradient here can be zero
                    combined_loss = pg_loss + v_loss * args.vf_coef
                    encoder_params = list(agent.network.parameters())
                    actor_head_params = _actor_head_params(agent)
                    critic_head_params = list(agent.critic.parameters())

                    def _grad_or_zeros(scalar_loss, params):
                        # Without a shared encoder one loss does not reach agent.network
                        grads = torch.autograd.grad(
                            scalar_loss,
                            params,
                            create_graph=False,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        return tuple(
                            torch.zeros_like(p) if g is None else g
                            for g, p in zip(grads, params)
                        )

                    with torch.no_grad():
                        grad_p_enc = _grad_or_zeros(pg_loss, encoder_params)
                        grad_v_enc = _grad_or_zeros(v_loss, encoder_params)
                        grad_tot_enc = _grad_or_zeros(combined_loss, encoder_params)
                        grad_p_act = _grad_or_zeros(pg_loss, actor_head_params)
                        grad_v_cri = _grad_or_zeros(v_loss, critic_head_params)
                        grad_p = torch.nn.utils.parameters_to_vector(grad_p_enc)
                        grad_v = torch.nn.utils.parameters_to_vector(grad_v_enc)
                        grad_tot = torch.nn.utils.parameters_to_vector(grad_tot_enc)
                        grad_p_act_vec = torch.nn.utils.parameters_to_vector(grad_p_act)
                        grad_v_cri_vec = torch.nn.utils.parameters_to_vector(grad_v_cri)
                        grad_p_similarity = torch.cosine_similarity(
                            grad_tot.unsqueeze(0), grad_p.unsqueeze(0)
                        ).item()
                        grad_v_similarity = torch.cosine_similarity(
                            grad_tot.unsqueeze(0), grad_v.unsqueeze(0)
                        ).item()
                        grad_policy_contribution = torch.dot(grad_tot, grad_p).item()
                        grad_value_contribution = torch.dot(grad_tot, grad_v).item()
                        inner_losses["policy_contribution"].append(
                            grad_policy_contribution
                        )
                        inner_losses["value_contribution"].append(
                            grad_value_contribution * args.vf_coef
                        )
                        value_contribution_fraction = abs(grad_value_contribution) / (
                            abs(grad_policy_contribution)
                            + abs(grad_value_contribution)
                            + 1e-8
                        )
                        inner_losses["value_contribution_fraction"].append(
                            value_contribution_fraction
                        )
                        grad_p_norm = grad_p.norm().item()
                        grad_v_norm = grad_v.norm().item()
                        grad_ratio = grad_p_norm / (grad_v_norm + grad_p_norm + 1e-8)
                        loss_ratio = pg_loss.abs().item() / (
                            v_loss.abs().item() + pg_loss.abs().item() + 1e-8
                        )
                        inner_losses["grad_v_similarity"].append(grad_v_similarity)
                        inner_losses["grad_p_similarity"].append(grad_p_similarity)
                        inner_losses["grad_ratio"].append(grad_ratio)
                        inner_losses["grad_p"].append(grad_p_norm)
                        inner_losses["grad_v"].append(grad_v_norm)
                        inner_losses["loss_ratio"].append(loss_ratio)

                        grad_v_scaled_norm = args.vf_coef * grad_v_norm
                        inner_losses["grad_v_scaled"].append(grad_v_scaled_norm)
                        inner_losses["grad_dominance_ratio"].append(
                            grad_v_scaled_norm / (grad_p_norm + 1e-12)
                        )
                        inner_losses["grad_alpha_est"].append(
                            grad_v_norm / (abs(v_loss.item()) + 1e-12)
                        )

                        grad_pv_cosine = torch.cosine_similarity(
                            grad_p.unsqueeze(0), grad_v.unsqueeze(0)
                        ).item()
                        inner_losses["grad_pv_cosine_encoder"].append(grad_pv_cosine)

                        inner_losses["grad_p_actor_head_norm"].append(
                            grad_p_act_vec.norm().item()
                        )
                        inner_losses["grad_v_critic_head_norm"].append(
                            grad_v_cri_vec.norm().item()
                        )

                        adam_snr_state["step"] += 1
                        inner_losses["adam_snr_pg_encoder"].append(
                            _update_adam_snr("pg_encoder", grad_p)
                        )
                        inner_losses["adam_snr_v_encoder"].append(
                            _update_adam_snr("v_encoder", grad_v)
                        )
                        inner_losses["adam_snr_pg_actor"].append(
                            _update_adam_snr("pg_actor", grad_p_act_vec)
                        )
                        inner_losses["adam_snr_v_critic"].append(
                            _update_adam_snr("v_critic", grad_v_cri_vec)
                        )

                        abs_g_tot = grad_tot.abs()
                        n_enc = abs_g_tot.numel()
                        inner_losses["grad_tot_enc_max"].append(abs_g_tot.max().item())
                        inner_losses["grad_tot_enc_p99"].append(
                            torch.quantile(abs_g_tot, 0.99).item()
                        )
                        inner_losses["grad_tot_enc_median"].append(
                            abs_g_tot.median().item()
                        )
                        inner_losses["grad_tot_enc_spike_ratio"].append(
                            (abs_g_tot.max() / (abs_g_tot.median() + 1e-12)).item()
                        )
                        sq = abs_g_tot.pow(2)
                        top_k = max(1, n_enc // 100)
                        topk_sq_sum = torch.topk(sq, top_k).values.sum()
                        inner_losses["grad_tot_enc_top1pct_mass"].append(
                            (topk_sq_sum / (sq.sum() + 1e-12)).item()
                        )

                    enc_params_before = [p.detach().clone() for p in encoder_params]
                if args.pcgrad:
                    # PCGrad on the (policy + entropy, value) split
                    l_pol = pg_loss - args.ent_coef * entropy_loss
                    l_val = v_loss * args.vf_coef
                    w_pol, w_val, pcgrad_stats = pcgrad_reweight(
                        l_pol, l_val, list(agent.parameters())
                    )
                    step_loss = w_pol * l_pol + w_val * l_val
                    if args.log_grad_stats:
                        for _k, _v in pcgrad_stats.items():
                            inner_losses[_k].append(_v)
                elif gradnorm is not None:
                    # GradNorm on the same (policy, value) split
                    l_pol = pg_loss - args.ent_coef * entropy_loss
                    l_val = v_loss * args.vf_coef
                    w_pol, w_val, gradnorm_stats = gradnorm.reweight(l_pol, l_val)
                    step_loss = w_pol * l_pol + w_val * l_val
                    for _k, _v in gradnorm_stats.items():
                        inner_losses[_k].append(_v)
                else:
                    step_loss = loss

                # Differentiable inner step, the agent depends on r_in_fn until stop_all_gradients
                inner_grad_norm = optim.step(step_loss, args.max_grad_norm)
                inner_losses["grad_norm"].append(inner_grad_norm.item())

                if args.use_popart:
                    # PopArt: update (mu, sigma), then rescale (W, b) to preserve outputs (Proposition 1)
                    agent.critic.update_stats(value_target.detach())
                    inner_losses["popart_mu"].append(agent.critic.mu.mean().item())
                    inner_losses["popart_sigma"].append(
                        agent.critic.sigma.mean().item()
                    )

                if args.log_grad_stats:
                    with torch.no_grad():
                        total_norm_val = inner_grad_norm.item()
                        clip_scale = min(
                            1.0,
                            args.max_grad_norm / (total_norm_val + 1e-12),
                        )
                        inner_losses["clip_scale"].append(clip_scale)

                        delta_parts = [
                            (p_after.detach() - p_before).flatten()
                            for p_after, p_before in zip(
                                agent.network.parameters(), enc_params_before
                            )
                        ]
                        delta_vec = torch.cat(delta_parts)
                        delta_norm = delta_vec.norm().item()
                        inner_losses["delta_theta_encoder"].append(delta_norm)

                        grad_p_norm_val = grad_p.norm().item()
                        grad_v_norm_val = grad_v.norm().item()
                        if delta_norm > 1e-12 and grad_p_norm_val > 1e-12:
                            delta_vs_grad_p = torch.cosine_similarity(
                                delta_vec.unsqueeze(0),
                                (-grad_p).unsqueeze(0),
                            ).item()
                        else:
                            delta_vs_grad_p = 0.0
                        if delta_norm > 1e-12 and grad_v_norm_val > 1e-12:
                            delta_vs_grad_v = torch.cosine_similarity(
                                delta_vec.unsqueeze(0),
                                (-grad_v).unsqueeze(0),
                            ).item()
                        else:
                            delta_vs_grad_v = 0.0
                        inner_losses["delta_theta_vs_neg_grad_p"].append(
                            delta_vs_grad_p
                        )
                        inner_losses["delta_theta_vs_neg_grad_v"].append(
                            delta_vs_grad_v
                        )

                inner_loss = loss.item()
                inner_pg_loss = pg_loss.item()
                inner_v_loss = v_loss.item()
                inner_entropy_loss = entropy_loss.item()
                inner_losses["loss"].append(inner_loss)
                inner_losses["pg_loss"].append(inner_pg_loss)
                inner_losses["v_loss"].append(inner_v_loss)
                inner_losses["entropy_loss"].append(inner_entropy_loss)
                inner_losses["advantages_max"].append(mb_advantages.max().item())
                inner_losses["advantages_mean"].append(mb_advantages.mean().item())
                inner_losses["advantages_std"].append(mb_advantages.std().item())
                inner_losses["advantages_min"].append(mb_advantages.min().item())
                inner_losses["values_max"].append(b_values_mix[mb_inds].max().item())
                inner_losses["values_mean"].append(b_values_mix[mb_inds].mean().item())
                inner_losses["values_std"].append(b_values_mix[mb_inds].std().item())
                inner_losses["values_min"].append(b_values_mix[mb_inds].min().item())

                if args.use_popart:
                    with torch.no_grad():
                        norm_newvalue = agent.critic.normalize(newvalue)
                        norm_target = agent.critic.normalize(mb_returns_mix)
                        norm_rollout = agent.critic.normalize(b_values_mix[mb_inds])

                        inner_losses["values_unnorm_max"].append(newvalue.max().item())
                        inner_losses["values_unnorm_mean"].append(
                            newvalue.mean().item()
                        )
                        inner_losses["values_unnorm_std"].append(newvalue.std().item())
                        inner_losses["values_unnorm_min"].append(newvalue.min().item())

                        inner_losses["values_norm_max"].append(
                            norm_newvalue.max().item()
                        )
                        inner_losses["values_norm_mean"].append(
                            norm_newvalue.mean().item()
                        )
                        inner_losses["values_norm_std"].append(
                            norm_newvalue.std().item()
                        )
                        inner_losses["values_norm_min"].append(
                            norm_newvalue.min().item()
                        )

                        inner_losses["returns_unnorm_max"].append(
                            mb_returns_mix.max().item()
                        )
                        inner_losses["returns_unnorm_mean"].append(
                            mb_returns_mix.mean().item()
                        )
                        inner_losses["returns_unnorm_std"].append(
                            mb_returns_mix.std().item()
                        )
                        inner_losses["returns_unnorm_min"].append(
                            mb_returns_mix.min().item()
                        )

                        inner_losses["returns_norm_max"].append(
                            norm_target.max().item()
                        )
                        inner_losses["returns_norm_mean"].append(
                            norm_target.mean().item()
                        )
                        inner_losses["returns_norm_std"].append(
                            norm_target.std().item()
                        )
                        inner_losses["returns_norm_min"].append(
                            norm_target.min().item()
                        )

                        inner_losses["values_rollout_norm_max"].append(
                            norm_rollout.max().item()
                        )
                        inner_losses["values_rollout_norm_mean"].append(
                            norm_rollout.mean().item()
                        )
                        inner_losses["values_rollout_norm_std"].append(
                            norm_rollout.std().item()
                        )
                        inner_losses["values_rollout_norm_min"].append(
                            norm_rollout.min().item()
                        )

                        residual = newvalue - b_values_mix[mb_inds]
                        residual_norm = norm_newvalue - norm_rollout
                        inner_losses["values_residual_unnorm_mean"].append(
                            residual.mean().item()
                        )
                        inner_losses["values_residual_unnorm_std"].append(
                            residual.std().item()
                        )
                        inner_losses["values_residual_norm_mean"].append(
                            residual_norm.mean().item()
                        )
                        inner_losses["values_residual_norm_std"].append(
                            residual_norm.std().item()
                        )

                do_meta = (args.meta_update and args.r_in_coef > 0) or use_practice
                if not do_meta:
                    stop_all_gradients(agent, optim, r_in_fn, r_in_optimizer)
                    continue

                # ------------------ Meta-update ---------------------
                # Outer extrinsic loss on the inner-updated agent, backpropagated into r_in_fn

                if use_practice:
                    if len(match_replay_buffer) < args.practice_batch_size:
                        stop_all_gradients(agent, optim, r_in_fn, r_in_optimizer)
                        continue

                    (
                        buf_obs,
                        buf_next_obs,
                        buf_actions,
                        buf_old_logprobs,
                        buf_rewards_ex,
                        buf_terms_ex,
                    ) = match_replay_buffer.sample(args.practice_batch_size)
                    buf_obs = buf_obs.to(device)
                    buf_next_obs = buf_next_obs.to(device)
                    buf_actions = buf_actions.to(device)
                    buf_old_logprobs = buf_old_logprobs.to(device)
                    buf_rewards_ex = buf_rewards_ex.to(device)
                    buf_terms_ex = buf_terms_ex.to(device)
                    buf_mode_flags = torch.ones(buf_obs.shape[0], device=device)

                    # Returns from the current v_ex, 1-step bootstrap zeroed on real termination
                    with torch.no_grad():
                        buf_next_values_ex = extrinsic_value(
                            agent,
                            r_in_fn,
                            buf_next_obs,
                            mode_flag=buf_mode_flags,
                            args=args,
                        ).view(-1)
                        buf_returns_ex = (
                            buf_rewards_ex
                            + args.gamma_outer
                            * buf_next_values_ex
                            * (1.0 - buf_terms_ex)
                        )

                    _, newlogprob, entropy, newvalue_agent, _ = (
                        agent.get_action_and_value(
                            buf_obs, buf_actions, mode_flag=buf_mode_flags
                        )
                    )
                    logratio = newlogprob - buf_old_logprobs
                    ratio = logratio.exp()

                    with torch.no_grad():
                        outer_old_approx_kl = (-logratio).mean()
                        outer_approx_kl = ((ratio - 1) - logratio).mean()
                        outer_clipfrac = (
                            ((ratio - 1.0).abs() > clipnow).float().mean().item()
                        )
                        outer_losses["approx_kl"].append(outer_approx_kl.item())
                        outer_losses["old_approx_kl"].append(outer_old_approx_kl.item())
                        outer_losses["clipfrac"].append(outer_clipfrac)
                        outer_losses["importance_ratio_mean"].append(
                            ratio.mean().item()
                        )
                        outer_losses["importance_ratio_max"].append(ratio.max().item())

                    # Advantages use a fresh v_ex baseline, as in Rajendran et al. (2020)
                    if args.practice_use_agent_vex:
                        newvalue = newvalue_agent.view(-1)
                    else:
                        newvalue = r_in_fn.get_extrinsic_value(
                            buf_obs, mode_flag=buf_mode_flags
                        ).view(-1)

                    mb_advantages = (buf_returns_ex - newvalue).detach()
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - clipnow, 1 + clipnow
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = 0.5 * ((newvalue - buf_returns_ex) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss + v_loss * args.vf_coef

                    r_in_optimizer.zero_grad()
                    loss.backward()
                    outer_grad_norm = nn.utils.clip_grad_norm_(
                        r_in_fn.parameters(), args.max_grad_norm
                    )
                    outer_losses["grad_norm"].append(outer_grad_norm.item())
                    r_in_optimizer.step()

                    outer_losses["loss"].append(loss.item())
                    outer_losses["pg_loss"].append(pg_loss.item())
                    outer_losses["v_loss"].append(v_loss.item())
                    outer_losses["entropy_loss"].append(entropy_loss.item())
                    outer_losses["advantages_max"].append(mb_advantages.max().item())
                    outer_losses["advantages_mean"].append(mb_advantages.mean().item())
                    outer_losses["advantages_std"].append(mb_advantages.std().item())
                    outer_losses["advantages_min"].append(mb_advantages.min().item())
                    outer_losses["values_max"].append(newvalue.max().item())
                    outer_losses["values_mean"].append(newvalue.mean().item())
                    outer_losses["values_std"].append(newvalue.std().item())
                    outer_losses["values_min"].append(newvalue.min().item())

                else:
                    outer_mb_inds = mb_inds
                    _, newlogprob, entropy, _, _ = agent.get_action_and_value(
                        b_obs[outer_mb_inds], b_actions[outer_mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[outer_mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        outer_old_approx_kl = (-logratio).mean()
                        outer_approx_kl = ((ratio - 1) - logratio).mean()
                        outer_clipfrac = (
                            ((ratio - 1.0).abs() > clipnow).float().mean().item()
                        )
                        outer_losses["approx_kl"].append(outer_approx_kl.item())
                        outer_losses["old_approx_kl"].append(outer_old_approx_kl.item())
                        outer_losses["clipfrac"].append(outer_clipfrac)

                    mb_advantages = b_advantages_ex[outer_mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - clipnow, 1 + clipnow
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = r_in_fn.get_extrinsic_value(b_obs[outer_mb_inds])
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns_ex[outer_mb_inds]) ** 2
                        v_clipped = b_values_ex[outer_mb_inds] + torch.clamp(
                            newvalue - b_values_ex[outer_mb_inds],
                            -clipnow,
                            clipnow,
                        )
                        v_loss_clipped = (v_clipped - b_returns_ex[outer_mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = (
                            0.5 * ((newvalue - b_returns_ex[outer_mb_inds]) ** 2).mean()
                        )

                    entropy_loss = entropy.mean()
                    loss = pg_loss + v_loss * args.vf_coef

                    r_in_optimizer.zero_grad()
                    loss.backward()
                    outer_grad_norm = nn.utils.clip_grad_norm_(
                        r_in_fn.parameters(), args.max_grad_norm
                    )
                    outer_losses["grad_norm"].append(outer_grad_norm.item())
                    r_in_optimizer.step()

                    outer_losses["loss"].append(loss.item())
                    outer_losses["pg_loss"].append(pg_loss.item())
                    outer_losses["v_loss"].append(v_loss.item())
                    outer_losses["entropy_loss"].append(entropy_loss.item())
                    outer_losses["advantages_max"].append(mb_advantages.max().item())
                    outer_losses["advantages_mean"].append(mb_advantages.mean().item())
                    outer_losses["advantages_std"].append(mb_advantages.std().item())
                    outer_losses["advantages_min"].append(mb_advantages.min().item())
                    outer_losses["values_max"].append(
                        b_values_ex[outer_mb_inds].max().item()
                    )
                    outer_losses["values_mean"].append(
                        b_values_ex[outer_mb_inds].mean().item()
                    )
                    outer_losses["values_std"].append(
                        b_values_ex[outer_mb_inds].std().item()
                    )
                    outer_losses["values_min"].append(
                        b_values_ex[outer_mb_inds].min().item()
                    )

                stop_all_gradients(agent, optim, r_in_fn, r_in_optimizer)

        # Polyak update of the target r_in network
        if args.use_target_rin and iteration % args.target_rin_update_freq == 0:
            with torch.no_grad():
                for p_target, p_live in zip(
                    r_in_target_fn.parameters(), r_in_fn.parameters()
                ):
                    p_target.mul_(1.0 - args.target_rin_tau).add_(
                        p_live, alpha=args.target_rin_tau
                    )
                for b_target, b_live in zip(
                    r_in_target_fn.buffers(), r_in_fn.buffers()
                ):
                    b_target.copy_(b_live)

        y_pred, y_true = b_values_mix.cpu().numpy(), b_returns_mix.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.log_grad_stats:
            with torch.no_grad():
                _kl_sum, _kl_n = 0.0, 0
                _vsq_sum, _v_n = 0.0, 0
                for _s in range(0, args.batch_size, args.minibatch_size):
                    _sl = slice(_s, _s + args.minibatch_size)
                    _mf = b_mode_flags[_sl] if use_practice else None
                    _, _, _, _v_new, _d_new = agent.get_action_and_value(
                        b_obs[_sl], mode_flag=_mf
                    )
                    _, _, _, _v_old, _d_old = _pre_update_agent.get_action_and_value(
                        b_obs[_sl], mode_flag=_mf
                    )
                    _kl = kl_divergence(_d_old, _d_new)
                    _kl_sum += _kl.sum().item()
                    _kl_n += _kl.numel()
                    _vsq_sum += (
                        (0.5 * (_v_new.view(-1) - _v_old.view(-1)) ** 2).sum().item()
                    )
                    _v_n += _v_new.view(-1).numel()
                _pk = "policy_phase" if args.use_ppg else "combined"
                inner_losses[f"phasic_reg/{_pk}_policy_drift"].append(
                    _kl_sum / max(_kl_n, 1)
                )
                inner_losses[f"phasic_reg/{_pk}_value_drift"].append(
                    _vsq_sum / max(_v_n, 1)
                )

        # DCPG value phase: every t_freq iterations, fit the value head on the buffer
        if args.use_ppg and iteration % args.t_freq == 0:
            old_agent = copy.deepcopy(agent)
            # Keep full rollout-batch references, the loop below rebinds b_obs / b_returns_mix
            rollout_obs = b_obs
            rollout_actions = b_actions
            rollout_logprobs = b_logprobs
            rollout_returns_mix = b_returns_mix
            rollout_values_mix = b_values_mix
            rollout_mode_flags = b_mode_flags
            for _ in range(args.k_v):
                ppg_sample = rb_ppg.sample()
                b_obs = ppg_sample[0].to(device)
                b_returns_mix = ppg_sample[1].to(device)
                ppg_mf = ppg_sample[2].to(device) if use_practice else None

                _, _, _, newvalue, dist = agent.get_action_and_value(
                    b_obs, mode_flag=ppg_mf
                )

                with torch.no_grad():
                    _, _, _, _, old_dist = old_agent.get_action_and_value(
                        b_obs, mode_flag=ppg_mf
                    )

                # Policy loss
                policy_loss = kl_divergence(old_dist, dist).mean()

                # Value loss (in normalized space under PopArt)
                newvalue = newvalue.view(-1)
                if args.use_popart:
                    pred = agent.critic.normalize(newvalue)
                    target = agent.critic.normalize(b_returns_mix)
                    v_loss = 0.5 * ((pred - target) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns_mix) ** 2).mean()

                loss = v_loss + args.beta_pi * policy_loss

                if args.log_grad_stats:
                    # DCPG gradient decomposition at the shared encoder, value phase
                    enc_params_v = list(agent.network.parameters())

                    _probe = torch.randint(
                        0, args.batch_size, (args.minibatch_size,), device=device
                    )
                    _probe_mf = rollout_mode_flags[_probe] if use_practice else None
                    _, _probe_logprob, _, _, _ = agent.get_action_and_value(
                        rollout_obs[_probe],
                        rollout_actions[_probe],
                        mode_flag=_probe_mf,
                    )
                    _probe_ratio = (_probe_logprob - rollout_logprobs[_probe]).exp()
                    _probe_adv = (
                        rollout_returns_mix[_probe] - rollout_values_mix[_probe]
                    )
                    if args.norm_adv:
                        _probe_adv = (_probe_adv - _probe_adv.mean()) / (
                            _probe_adv.std() + 1e-8
                        )
                    _probe_pg = torch.max(
                        -_probe_adv * _probe_ratio,
                        -_probe_adv
                        * torch.clamp(_probe_ratio, 1 - clipnow, 1 + clipnow),
                    ).mean()

                    with torch.no_grad():
                        gv_vec = _encoder_grad_vec(v_loss, enc_params_v)
                        gk_vec = _encoder_grad_vec(policy_loss, enc_params_v)
                        gpi_vec = _encoder_grad_vec(_probe_pg, enc_params_v)
                        gtot_vec = _encoder_grad_vec(loss, enc_params_v)
                        gv_n = gv_vec.norm().item()
                        gk_n = gk_vec.norm().item()
                        inner_losses["v/grad_v_encoder"].append(gv_n)
                        inner_losses["v/grad_kl_encoder"].append(gk_n)
                        inner_losses["v/grad_pi_encoder"].append(gpi_vec.norm().item())
                        inner_losses["v/grad_kl_ratio"].append(
                            (args.beta_pi * gk_n) / (gv_n + 1e-12)
                        )
                        inner_losses["v/grad_v_kl_cosine"].append(_cos(gv_vec, gk_vec))
                        for _k, (_a, _b) in {
                            "cos_g_gpi": (gtot_vec, gpi_vec),
                            "cos_g_gV": (gtot_vec, gv_vec),
                            "cos_g_gKL": (gtot_vec, gk_vec),
                            "cos_gV_gKL": (gv_vec, gk_vec),
                            "cos_gpi_gKL": (gpi_vec, gk_vec),
                            "cos_gV_gpi": (gv_vec, gpi_vec),
                        }.items():
                            inner_losses[f"phasic_val/{_k}"].append(_cos(_a, _b))

                optim.step(loss, args.max_grad_norm)

                torchopt.stop_gradient(agent)
                torchopt.stop_gradient(optim)
                torchopt.stop_gradient(r_in_fn)

                inner_losses["v/policy_loss"].append(policy_loss.item())
                inner_losses["v/value_loss"].append(v_loss.item())
                inner_losses["v/loss"].append(loss.item())
                inner_losses["v/values_max"].append(newvalue.max().item())
                inner_losses["v/values_mean"].append(newvalue.mean().item())
                inner_losses["v/values_min"].append(newvalue.min().item())

            if args.log_grad_stats:
                with torch.no_grad():
                    _kl_sum, _kl_n = 0.0, 0
                    _vsq_sum, _v_n = 0.0, 0
                    for _s in range(0, args.batch_size, args.minibatch_size):
                        _sl = slice(_s, _s + args.minibatch_size)
                        _mf = rollout_mode_flags[_sl] if use_practice else None
                        _, _, _, _v_new, _d_new = agent.get_action_and_value(
                            rollout_obs[_sl], mode_flag=_mf
                        )
                        _, _, _, _v_old, _d_old = old_agent.get_action_and_value(
                            rollout_obs[_sl], mode_flag=_mf
                        )
                        _kl = kl_divergence(_d_old, _d_new)
                        _kl_sum += _kl.sum().item()
                        _kl_n += _kl.numel()
                        _vsq_sum += (
                            (0.5 * (_v_new.view(-1) - _v_old.view(-1)) ** 2)
                            .sum()
                            .item()
                        )
                        _v_n += _v_new.view(-1).numel()
                    inner_losses["phasic_reg/value_phase_policy_drift"].append(
                        _kl_sum / max(_kl_n, 1)
                    )
                    inner_losses["phasic_reg/value_phase_value_drift"].append(
                        _vsq_sum / max(_v_n, 1)
                    )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        print("SPS:", int(global_step / (time.time() - start_time)), flush=True)
        memory_usage = process.memory_info().rss / 1024**2
        print(f"Memory Usage (MB): {memory_usage}", flush=True)
        if writer:
            writer.add_scalar("charts/learning_rate", lrnow, global_step)
            for key, value in inner_losses.items():
                writer.add_scalar(f"losses/inner_{key}", np.mean(value), global_step)
            for key, value in outer_losses.items():
                writer.add_scalar(f"losses/outer_{key}", np.mean(value), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )
            log_stats("r_ex", rewards_ex)
            r_ex_density = (rewards_ex != 0).float().mean().item()
            writer.add_scalar("charts/r_ex_density", r_ex_density, global_step)
            log_stats("r_in", rewards_in)
            reward_similarity = torch.cosine_similarity(
                rewards_ex.reshape(-1).unsqueeze(0),
                rewards_in.reshape(-1).unsqueeze(0),
            ).item()
            writer.add_scalar(
                "charts/ext_int_reward_similarity", reward_similarity, global_step
            )
            writer.add_scalar(
                "charts/frac_r_in_saturated",
                (rewards_in.abs() > 0.99).float().sum().item()
                / ((rewards_in != 0).float().sum().item() + 1e-8),
                global_step,
            )
            log_stats("r_rnd", rewards_rnd, stats=("mean", "max", "min", "std"))
            writer.add_scalar("charts/mean_mse_rnd", mse_rnd.mean().item(), global_step)
            writer.add_scalar(
                "charts/r_in_coef", args.r_in_coef * r_in_multiplier, global_step
            )
            writer.add_scalar("charts/r_ex_coef", args.r_ex_coef, global_step)
            writer.add_scalar("charts/r_rnd_coef", args.r_rnd_coef, global_step)
            writer.add_scalar("charts/r_in_multiplier", r_in_multiplier, global_step)
            writer.add_scalar("charts/memory_usage", memory_usage, global_step)
            if use_practice:
                practice_frac = (mode_flags == 0).float().mean().item()
                match_frac = (mode_flags == 1).float().mean().item()
                writer.add_scalar("charts/practice_frac", practice_frac, global_step)
                writer.add_scalar("charts/match_frac", match_frac, global_step)
                writer.add_scalar(
                    "charts/n_practice_envs",
                    (match_tags == 0).sum().item(),
                    global_step,
                )
                writer.add_scalar(
                    "charts/n_match_envs",
                    (match_tags == 1).sum().item(),
                    global_step,
                )
                writer.add_scalar(
                    "charts/practice_buffer_size",
                    len(match_replay_buffer),
                    global_step,
                )
                # Per-mode reward stats
                match_mask_2d = mode_flags == 1
                practice_mask_2d = mode_flags == 0
                if match_mask_2d.any():
                    writer.add_scalar(
                        "charts/mean_r_ex_match",
                        rewards_ex[match_mask_2d].mean().item(),
                        global_step,
                    )
                if practice_mask_2d.any():
                    log_stats(
                        "r_in_practice",
                        rewards_in[practice_mask_2d],
                        stats=("mean", "std"),
                    )

        if args.checkpoint_every_iteration:
            practice_state = None
            if use_practice:
                practice_state = {
                    "match_tags": match_tags.detach().cpu(),
                    "practice_step_counts": practice_step_counts.copy(),
                    "match_replay_buffer": match_replay_buffer,
                }
            save_checkpoint(
                os.path.join(args.runs_dir, run_name, "checkpoint.pt"),
                args=args,
                iteration=iteration,
                global_step=global_step,
                match_step=match_step,
                run_name=run_name,
                elapsed_time=time.time() - start_time,
                return_timesteps=return_timesteps,
                delay_r_ex=delay_r_ex,
                delay_step=delay_step,
                next_obs=next_obs,
                next_done=next_done,
                obs_mean=obs_mean,
                obs_std=obs_std,
                agent=agent,
                r_in_fn=r_in_fn,
                r_in_target_fn=r_in_target_fn,
                r_rnd_fn=r_rnd_fn,
                optim=optim,
                r_in_optimizer=r_in_optimizer,
                r_rnd_optimizer=r_rnd_optimizer,
                rb_ppg=rb_ppg,
                envs=envs,
                adam_snr_state=adam_snr_state,
                practice_state=practice_state,
                gradnorm=gradnorm,
            )

    if args.save_agent_base_model:
        policy_encoder, value_encoder = _agent_policy_value_encoders(agent)
        path = os.path.join(args.runs_dir, run_name, "actor_network.pt")
        torch.save(policy_encoder.state_dict(), path)
        path = os.path.join(args.runs_dir, run_name, "critic_network.pt")
        torch.save(value_encoder.state_dict(), path)
        path = os.path.join(args.runs_dir, run_name, "agent.pt")
        torch.save(agent.state_dict(), path)
    if args.save_rin_base_model:
        path = os.path.join(args.runs_dir, run_name, "rin_base.pt")
        torch.save(r_in_fn.base_model.state_dict(), path)
        path = os.path.join(args.runs_dir, run_name, "rin.pt")
        torch.save(r_in_fn.state_dict(), path)

    envs.close()


if __name__ == "__main__":
    # Parse once to read env_id, then again with that domain's default hyperparameters.
    args, _ = tyro.cli(Args, return_unknown_args=True)
    if "NoFrameskip" in args.env_id:
        args = tyro.cli(AtariArgs)
    elif "procgen" in args.env_id:
        args = tyro.cli(ProcGenArgs)
    else:
        args = tyro.cli(MuJoCoArgs)
    if args.checkpoint_load_path:
        # Loaded here to reuse its run_name, train() reads it from _LOADED_CHECKPOINT
        _LOADED_CHECKPOINT = torch.load(
            args.checkpoint_load_path, map_location="cpu", weights_only=False
        )
        run_name = _LOADED_CHECKPOINT["run_name"]
        print(
            f"Resuming run_name={run_name} from {args.checkpoint_load_path} "
            f"(last completed iteration: {_LOADED_CHECKPOINT['iteration']})",
            flush=True,
        )
    else:
        run_name = f"{os.path.basename(args.env_id)}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            group=os.path.basename(args.runs_dir),
            monitor_gym=True,
            save_code=True,
        )
    hyperparams_md = "|param|value|\n|-|-|\n%s" % (
        "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
    )
    print(hyperparams_md, flush=True)
    if args.runs_dir:
        writer = SummaryWriter(os.path.join(args.runs_dir, run_name))
        writer.add_text(
            "hyperparameters",
            hyperparams_md,
        )
    else:
        writer = None
    train_with_logging(args, train_fn=train, writer=writer)
