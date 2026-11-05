from dataclasses import dataclass
from typing import Literal


@dataclass
class Args:
    exp_name: str = "ppo_rin"
    """the name of this experiment"""
    runs_dir: str = "runs-atari"
    """the directory to save the experiment results"""
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "lirpg-instability"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_agent_base_model: bool = True
    """whether to save the agent encoders into the `{runs_dir}/{run_name}` folder"""
    save_rin_base_model: bool = True
    """whether to save the intrinsic reward function into the `{runs_dir}/{run_name}` folder"""
    agent_base_model_path: str = ""
    """path to an agent encoder state_dict to initialize from"""
    agent_base_model_target: Literal["all", "policy"] = "all"
    """which encoder(s) `agent_base_model_path` loads into"""
    checkpoint_every_iteration: bool = False
    """if toggled, save a resumable checkpoint at the end of every iteration"""
    checkpoint_load_path: str = ""
    """path to a checkpoint.pt to resume training from"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    r_in_learning_rate: float = 2.5e-4
    """the learning rate of the intrinsic reward function optimizer"""
    r_rnd_learning_rate: float = 1e-3
    """the learning rate of the RND predictor optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma of the inner (mixed-reward) return"""
    gamma_outer: float = 0.99
    """the discount factor for the outer (extrinsic) return"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    gae_lambda_outer: float = 0.95
    """the lambda for the outer general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    rnd_update_epochs: int = 4
    """the epochs to update the RND predictor"""
    norm_adv: bool = True
    """toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """toggles whether or not to use a clipped loss for the value function"""
    use_target_rin: bool = False
    """if toggled, value targets use a slowly updated target copy of r_in (not with use_ppg)"""
    target_rin_tau: float = 0.05
    """the Polyak coefficient of the target r_in update"""
    target_rin_update_freq: int = 1
    """the number of iterations between target r_in updates"""
    r_ex_coef: float = 1
    """coefficient of the extrinsic reward"""
    r_in_coef: float = 0
    """coefficient of the learned intrinsic reward"""
    r_rnd_coef: float = 0
    """coefficient of the RND intrinsic reward"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    r_in_activation: Literal["tanh", "relu"] = "tanh"
    """the output activation of the intrinsic reward function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    meta_update: bool = False
    """toggles the meta-update of the intrinsic reward function"""
    reward_freq: int = 1
    """deliver the accumulated reward every this many steps (1 = dense)"""
    rnd_warmup_steps: int = 6400
    """the number of random steps used to initialize the RND observation statistics"""
    random_policy: bool = False
    """if toggled, act with a uniformly random policy"""
    handle_sparse_rex: Literal["none", "r_in_if_r_ex", "scale_r_in_by_density"] = "none"
    """how r_in is gated when r_ex is sparse"""
    shared_networks: str = "rin-vex,pi-vmix"
    """comma-separated encoder-sharing pairs, any of 'rin-vex' and 'pi-vmix' ('' shares none)"""
    log_grad_stats: bool = False
    """if toggled, log gradient statistics"""

    pcgrad: bool = False
    """if toggled, apply PCGrad to the policy/value update (not with use_ppg)"""

    gradnorm: bool = False
    """if toggled, apply GradNorm to the policy/value update (not with pcgrad or use_ppg)"""
    gradnorm_alpha: float = 0
    """the GradNorm asymmetry hyperparameter alpha"""
    gradnorm_lr: float = 2.5e-4
    """the learning rate of the GradNorm task-weight optimizer"""

    use_popart: bool = False
    """if toggled, the inner value head uses PopArt value normalization"""
    popart_beta: float = 3e-4
    """the EMA rate of the PopArt value statistics"""

    use_ppg: bool = False
    """if toggled, decouple policy and value updates with DCPG"""
    t_off: int = 16
    """the DCPG buffer size, in rollouts"""
    t_freq: int = 32
    """number of iterations between two consecutive DCPG value phases"""
    k_v: int = 64
    """number of value minibatch updates per value phase"""
    beta_pi: float = 1
    """value-phase policy regularization strength"""
    beta_v: float = 0
    """policy-phase value regularization strength"""
    m_v: int = 256
    """value minibatch size (states per minibatch, sampled with replacement)"""

    freeze_rin_frac: float = 1.0
    """fraction of training after which the intrinsic reward function is frozen"""

    # Practice-match arguments (Rajendran et al. 2020)
    practice_steps: int = 0
    """the number of practice steps per match phase (0 disables practice-match)"""
    practice_buffer_size: int = 12000
    """the capacity of the match replay buffer"""
    practice_batch_size: int = 1000
    """the meta-update batch size sampled from the match replay buffer"""
    practice_use_agent_vex: bool = False
    """if toggled, the meta-update uses the agent's match-mode value head as v_ex"""
    max_match_timesteps: int | None = None
    """if set, stop after this many match-mode env steps instead of `total_timesteps`"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    handle_trunc: bool = True
    """if toggled, bootstrap from the final observation's value on truncated episodes"""


@dataclass
class AtariArgs(Args):
    env_id: str = "BreakoutNoFrameskip-v4"
    total_timesteps: int = 10_000_000
    num_envs: int = 8
    num_steps: int = 128
    r_in_learning_rate: float = 2.5e-4
    r_rnd_learning_rate: float = 1e-3
    k_v: int = 256


@dataclass
class MuJoCoArgs(Args):
    env_id: str = "HalfCheetah-v4"
    runs_dir: str = "runs-mujoco"
    total_timesteps: int = 1_000_000
    num_envs: int = 1
    num_steps: int = 2048
    learning_rate: float = 3e-4
    r_in_learning_rate: float = 1e-4
    r_rnd_learning_rate: float = 1e-3
    anneal_lr: bool = False
    num_minibatches: int = 32
    update_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    reward_freq: int = 20
    shared_networks: str = ""


# PPO hyperparameters from CleanRL ppo_procgen.py
@dataclass
class ProcGenArgs(Args):
    env_id: str = "procgen-starpilot"
    """procgen env id in the format 'procgen-<game_name>'"""
    runs_dir: str = "runs-procgen"
    total_timesteps: int = 25_000_000
    num_envs: int = 64
    num_steps: int = 256
    learning_rate: float = 5e-4
    r_in_learning_rate: float = 5e-4
    r_rnd_learning_rate: float = 1e-3
    anneal_lr: bool = False
    num_minibatches: int = 8
    update_epochs: int = 3
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    gamma: float = 0.999
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    shared_networks: str = "rin-vex,pi-vmix"

    num_levels: int = 200
    """number of unique levels to train on (0 = unlimited)"""
    start_level: int = 0
    """starting level seed"""
    distribution_mode: str = "easy"
    """difficulty distribution mode (easy, hard, exploration, memory, extreme)"""
