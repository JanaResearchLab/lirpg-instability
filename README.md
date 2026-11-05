# Instability of Meta-Learning Intrinsic Rewards for Policy Gradient Reinforcement Learning

This repository is the official implementation of *Instability of Meta-Learning Intrinsic Rewards for Policy Gradient Reinforcement Learning* (Dilith Jayakody, Domenic Rosati, Janarthanan Rajendran; NeurIPS 2026).

## Requirements

Each domain needs its own Python 3.10 environment, because the Atari, MuJoCo and ProcGen simulators pin different versions of gym, gymnasium and torch.

| Environment | Requirements | Used for |
|---|---|---|
| `.venv_atari` | `requirements/atari.txt` + `requirements/plotting.txt` | Atari training, all figures and tables |
| `.venv_mujoco` | `requirements/mujoco.txt` | MuJoCo training |
| `.venv_procgen` | `requirements/procgen.txt` | ProcGen training |

`setup_env.sh` creates all three. It needs Python 3.10 (set `PYTHON=/path/to/python3.10` if `python3.10` is not on your path). For Atari it also installs the plotting requirements and the ROMs (`AutoROM --accept-license`). On a Compute Canada / DRAC cluster it loads the environment modules the paper's runs used and takes numpy, scipy and opencv from them.

```setup
bash setup_env.sh            # or: bash setup_env.sh atari
```

`requirements/freeze/` lists every package and version in the environments used for the paper. These were built on Compute Canada, so some versions carry a `+computecanada` suffix. Those wheels come from the Compute Canada wheelhouse and are not on PyPI. Elsewhere, use the top-level `requirements/*.txt` files, or drop the suffix, which usually gives the matching PyPI release.

`torchopt_patch.py` adds gradient-norm clipping to torchopt's differentiable `MetaOptimizer.step`. It is applied automatically when `ppo_rin.py` is imported and was written for torchopt 0.7.3. The clipping rescales the gradients outside autograd, so the meta-gradient through the inner update does not include the clip factor. The paper's runs used this behaviour.

## Training

`ppo_rin.py` trains one agent. Every flag and its default is documented in `configs.py`; the domain (and its default hyperparameters, Table 1 of the paper) is picked from `--env_id`. Examples:

```train
# PPO (extrinsic only) on Breakout
python ppo_rin.py --env_id BreakoutNoFrameskip-v4 --seed 0 --r_ex_coef 1 --r_in_coef 0 --no-meta_update

# Intrinsic-only PPO-LIRPG, without and with DCPG
python ppo_rin.py --env_id BreakoutNoFrameskip-v4 --seed 0 --r_ex_coef 0 --r_in_coef 1 --meta_update
python ppo_rin.py --env_id BreakoutNoFrameskip-v4 --seed 0 --r_ex_coef 0 --r_in_coef 1 --meta_update \
    --use_ppg --t_freq 32 --t_off 16 --k_v 256 --m_v 256 --beta_pi 1 --beta_v 0

# Mixed-reward PPO-LIRPG on MuJoCo (lambda_in = 0.3, Table 5)
python ppo_rin.py --env_id InvertedPendulum-v4 --seed 0 --r_ex_coef 0.7 --r_in_coef 0.3 --meta_update
```

The main flags:

| Flag | Meaning |
|---|---|
| `--r_in_coef`, `--r_ex_coef` | $\lambda_\mathrm{in}$ and $1-\lambda_\mathrm{in}$ in the mixed reward (Eq. 1) |
| `--meta_update` | meta-learn $r^\mathrm{in}$ (LIRPG); off for PPO |
| `--use_ppg --t_freq --t_off --k_v --m_v --beta_pi --beta_v` | DCPG and its hyperparameters (Table 6; Atari defaults `k_v=256`, ProcGen `k_v=64`) |
| `--shared_networks` | which networks share an encoder (`rin-vex`, `pi-vmix`; empty for fully separate encoders) |
| `--handle_sparse_rex`, `--vf_coef`, `--gamma`, `--use_popart`, `--use_target_rin`, `--pcgrad`, `--gradnorm` | the alternate strategies of Section 7 |
| `--practice_steps`, `--practice_buffer_size`, `--practice_use_agent_vex`, `--max_match_timesteps` | practice-match (Appendix B.5) |
| `--r_rnd_coef` | RND bonus (Appendix E.11) |
| `--log_grad_stats` | log the encoder gradient measurements (Section 5, Appendix D) |
| `--checkpoint_every_iteration`, `--checkpoint_load_path` | save full training state each iteration and resume a run exactly |

Each run writes TensorBoard logs, and the saved agent and $r^\mathrm{in}$ networks, to `<runs_dir>/<env_id>__<exp_name>__<seed>__<timestamp>/`.

### Reproducing the paper's experiments

Each script in `experiments/` lists the runs behind one or more results, with the same run directories and experiment names the plotting code expects. A script prints one training command per run. First save them to a file:

```train
bash experiments/dcpg.sh > commands.txt
```

Then run the file in one of three ways:

```train
bash commands.txt                          # one after another
parallel -j 8 < commands.txt               # 8 at a time, with GNU parallel
bash slurm/submit_array.sh commands.txt    # as SLURM job arrays (set SBATCH_ACCOUNT first)
```

Run the commands from the repository root, in the environment of their domain. `slurm/submit_array.sh` splits a command file by domain and loads the right environment for each.

| Paper result | Script | Runs |
|---|---|---|
| Figures 1-3, 6, 11, 17, Tables 4-5 (LIRPG) | `experiments/lambda_sweep.sh` | 420 |
| Figure 4 | `experiments/frozen_rin.sh` | 2 |
| Figure 5 | `experiments/shared_encoders.sh` | 126 |
| Appendix B.3 (held-out mixed reward) | `experiments/dcpg.sh` | 42 |
| Figures 6, 8, Table 4 (LIRPG + DCPG) | `experiments/dcpg_lambda_sweep.sh` | 168 |
| Figures 7(a), 12, 13 | `experiments/atari49.sh` | 567 |
| Figures 7(b), 14 | `experiments/procgen.sh` | 45 |
| Figures 7(c), 15, 16 | `experiments/practice_match.sh` | 24 |
| Figure 9 | `experiments/alternate_strategies.sh` | 168 |
| Figures 6, 10, 17 | `experiments/dcpg_regularization.sh` | 84 |
| Tables 8, 13 | `experiments/grad_stats.sh` | 60 |
| Table 9 | `experiments/phase_drift.sh` | 21 |
| Table 10 | `experiments/pretrained_encoder.sh` (after `lambda_sweep.sh`) | 21 |
| Table 11 | `experiments/tfreq_sweep.sh` | 60 |
| Table 12 | `experiments/vf_coef_sweep.sh` | 75 |
| Appendix E.5 | `experiments/relu_activation.sh` | 6 |
| Figure 18 | `experiments/rnd.sh` | 9 |

All results use 3 seeds (0-2), except Figure 4 (seed 0 only) and the held-out check of Appendix B.3 (seeds 10-12). Per-run wall-clock times and the total compute are in Appendix B.1 (Table 2): an Atari PPO-LIRPG run takes about 12 hours on a shared GPU node, MuJoCo about 8 hours on 2 CPU cores, and ProcGen about 7 hours on one A100.

## Evaluation

Figures and tables are computed from the TensorBoard logs; there is no separate evaluation phase. Scores are the mean episodic return over the last 100 training episodes, normalized per game as $`(\mathrm{score} - \mathrm{score}_\mathrm{random}) / (\mathrm{score}_\mathrm{max} - \mathrm{score}_\mathrm{random})`$, where $`\mathrm{score}_\mathrm{random}`$ is the median random-policy run and $`\mathrm{score}_\mathrm{max}`$ is the best seed of any run that a figure or table of the paper reports on that game. The floors and ceilings of the paper's runs are in `plotting/score_bounds.json`. Keep this file to compare your runs with the paper. The stored floor is used as is, and each game's ceiling is the larger of the stored value and the best of your runs. `python plotting/paper_figures.py bounds` raises the stored ceiling where one of your runs beats it, replaces the floor with your random-policy runs, and leaves games without runs unchanged. Aggregates use rliable with 95% stratified-bootstrap confidence intervals. Run in `.venv_atari`, from the repository root:

```eval
bash plotting/make_figures.sh            # all figures whose runs exist
bash plotting/make_figures.sh fig6       # one figure (see the function names in the script)
python plotting/paper_tables.py table8   # tables: table4, table5, table8, table9, table10, table11, table12, table13
```

`plotting/paper_figures.py` writes Figures 1, 2, 5, 6, 7, 10, 15 and 16 and the separate legend files to `figures/`, with the paper's file names and layout, and prints the held-out comparison of Appendix B.3. The other figures are written next to the runs, in `<runs_dir>/figures*/`, except Figure 17, which goes to `plots/r_in_states/`. Figure 17 reads the agent (`agent.pt`) and intrinsic reward network (`rin.pt`) that every run saves. `plotting/visualize_hyperparameters.py` (the `fig3` step) also writes `<runs_dir>/best_runs.json`, the per-game best $\lambda_\mathrm{in}$ among the candidates of Appendix B.3, which `paper_tables.py table4` and `table5` read. Runs with fewer than 100 episodes are skipped with a warning, and aggregates then use the smallest seed count of any game. The mixed-reward coefficients used by the experiment scripts are those of Table 4 and are fixed in `experiments/common.sh`.

## Code layout

| File | Contents |
|---|---|
| `ppo_rin.py` | `train()`: rollouts, the inner PPO update (made differentiable with torchopt's `MetaAdam`), the LIRPG meta-update, the DCPG value phase, practice-match, and logging |
| `configs.py` | all command-line flags and the per-domain defaults |
| `agents.py` | policy/value networks per domain, and the PopArt value head |
| `intrinsic_reward_module.py` | the learned intrinsic reward and extrinsic value networks, and RND |
| `envs.py` | environment construction and wrappers per domain |
| `buffers.py` | the DCPG value-phase buffer and the practice-match replay buffer |
| `checkpointing.py` | full-state checkpointing for exact resume |
| `torchopt_patch.py` | gradient clipping in torchopt's differentiable step |
| `experiments/`, `slurm/` | the paper's runs and a SLURM launcher |
| `plotting/` | figures and tables |
| `tests/test_smoke.py` | a few iterations of each configuration: `pytest tests/test_smoke.py` |

## Acknowledgements

The PPO implementation builds on [CleanRL](https://github.com/vwxyzjn/cleanrl) (MIT License), including its ProcGen IMPALA encoder, which CleanRL adapted from the [AIcrowd ProcGen starter kit](https://github.com/AIcrowd/neurips2020-procgen-starter-kit) (Apache 2.0). Their notices are in `licenses/cleanrl-LICENSE.txt`. The RND running statistics follow the [reference implementation](https://github.com/openai/random-network-distillation). Aggregate metrics use [rliable](https://github.com/google-research/rliable).

## License

MIT; see `LICENSE`. Third-party code is noted in the source files, with its licenses in `licenses/`.

## Citation

```bibtex
@inproceedings{jayakody2026instability,
  title     = {Instability of Meta-Learning Intrinsic Rewards for Policy Gradient Reinforcement Learning},
  author    = {Jayakody, Dilith and Rosati, Domenic and Rajendran, Janarthanan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
