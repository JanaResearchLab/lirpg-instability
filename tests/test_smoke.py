"""Short CPU runs of each paper configuration; tests whose simulator is not installed are skipped.

Usage: pytest tests/test_smoke.py [-k atari]
"""

import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INT = "--r_ex_coef 0 --r_in_coef 1 --meta_update"
DCPG = "--use_ppg --t_freq 2 --k_v 4"
PRACTICE = (
    "--r_ex_coef 1 --r_in_coef 1 --meta_update --practice_steps 20 --practice_buffer_size 500 "
    "--practice_batch_size 64 --practice_use_agent_vex"
)

ATARI = (
    "--env_id BreakoutNoFrameskip-v4 --num_envs 2 --num_steps 64 --total_timesteps 512"
)
MUJOCO = "--env_id InvertedPendulum-v4 --num_steps 256 --num_minibatches 4 --total_timesteps 1024"
PROCGEN = "--env_id procgen-coinrun --num_envs 2 --num_steps 64 --num_minibatches 2 --total_timesteps 512"

CASES = {
    "atari-ppo": (ATARI, "--r_ex_coef 1 --r_in_coef 0 --no-meta_update"),
    "atari-lirpg": (ATARI, INT),
    "atari-lirpg-mixed-unshared": (
        ATARI,
        "--r_ex_coef 0.9 --r_in_coef 0.1 --meta_update --shared_networks=",
    ),
    "atari-lirpg-dcpg-gradstats": (ATARI, f"{INT} {DCPG} --log_grad_stats"),
    "atari-alternates": (
        ATARI,
        f"{INT} --use_popart --handle_sparse_rex scale_r_in_by_density",
    ),
    "atari-target-rin": (ATARI, f"{INT} --use_target_rin --target_rin_tau 0.02"),
    "atari-pcgrad": (ATARI, f"{INT} --pcgrad"),
    "atari-gradnorm": (ATARI, f"{INT} --gradnorm"),
    "atari-practice-dcpg": (ATARI, f"{PRACTICE} {DCPG}"),
    "atari-rnd": (
        ATARI,
        "--r_ex_coef 0.9 --r_in_coef 0 --r_rnd_coef 0.1 --no-meta_update --rnd_warmup_steps 64",
    ),
    "mujoco-lirpg": (MUJOCO, INT),
    "mujoco-lirpg-relu": (MUJOCO, f"{INT} --r_in_activation relu"),
    "procgen-lirpg-dcpg": (PROCGEN, f"{INT} {DCPG}"),
}
REQUIRES = {"atari": "ale_py", "mujoco": "mujoco", "procgen": "procgen"}


@pytest.mark.parametrize("name", sorted(CASES))
def test_config_runs(name, tmp_path):
    module = REQUIRES[name.split("-")[0]]
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not installed in this environment")
    common, flags = CASES[name]
    cmd = (
        f"{sys.executable} ppo_rin.py {common} {flags} --seed 1 --no-cuda --no-track "
        f"--runs_dir {tmp_path} --exp_name=smoke"
    )
    out = subprocess.run(cmd.split(), cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    assert "An exception occurred during training" not in out.stdout
