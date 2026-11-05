"""Save and restore the full training state so a resumed run matches an uninterrupted one."""

import os
import pickle
import random
import time

import numpy as np
import torch

from envs import ProcGenVecEnv


def _capture_rng_state():
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(state):
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _try_pickle(obj, label):
    try:
        return pickle.dumps(obj)
    except Exception as e:
        print(
            f"Warning: could not pickle {label} for checkpointing: {e!r}. "
            "Resume will restart this component from a fresh state.",
            flush=True,
        )
        return None


def save_checkpoint(
    path,
    *,
    args,
    iteration,
    global_step,
    match_step,
    run_name,
    elapsed_time,
    return_timesteps,
    delay_r_ex,
    delay_step,
    next_obs,
    next_done,
    obs_mean,
    obs_std,
    agent,
    r_in_fn,
    r_rnd_fn,
    optim,
    r_in_optimizer,
    r_rnd_optimizer,
    rb_ppg,
    envs,
    r_in_target_fn=None,
    adam_snr_state=None,
    practice_state=None,
    gradnorm=None,
):
    ckpt = {
        "iteration": iteration,
        "global_step": global_step,
        "match_step": match_step,
        "run_name": run_name,
        "elapsed_time": elapsed_time,
        "return_timesteps": return_timesteps,
        "delay_r_ex": delay_r_ex,
        "delay_step": delay_step,
        "next_obs": next_obs.detach().cpu(),
        "next_done": next_done.detach().cpu(),
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "adam_snr_state": adam_snr_state,
        "gradnorm_state": gradnorm.state_dict() if gradnorm is not None else None,
        "mutable_args": {
            "freeze_rin_frac": args.freeze_rin_frac,
            "meta_update": args.meta_update,
        },
        "agent_state_dict": agent.state_dict(),
        "r_in_fn_state_dict": r_in_fn.state_dict(),
        "r_in_target_fn_state_dict": (
            r_in_target_fn.state_dict() if r_in_target_fn is not None else None
        ),
        "r_rnd_fn_state_dict": r_rnd_fn.state_dict(),
        "r_rnd_extras": {
            "obs_rms_mean": r_rnd_fn.obs_rms.mean,
            "obs_rms_var": r_rnd_fn.obs_rms.var,
            "obs_rms_count": r_rnd_fn.obs_rms.count,
            "rew_rms_mean": r_rnd_fn.rew_rms.mean,
            "rew_rms_var": r_rnd_fn.rew_rms.var,
            "rew_rms_count": r_rnd_fn.rew_rms.count,
            "discounted_rewems": r_rnd_fn.discounted_reward.rewems,
        },
        "optim_state_dict": optim.state_dict(),
        "r_in_optimizer_state_dict": r_in_optimizer.state_dict(),
        "r_rnd_optimizer_state_dict": r_rnd_optimizer.state_dict(),
        "rb_ppg_pickle": (
            _try_pickle(rb_ppg, "PPGReplayBuffer") if rb_ppg is not None else None
        ),
        "practice_state": practice_state,
        "rng_state": _capture_rng_state(),
    }

    # ProcGen's C++ env is not picklable but has its own get_state/set_state.
    if isinstance(envs, ProcGenVecEnv):
        ckpt["procgen_state"] = envs.get_state()
        ckpt["envs_pickle"] = None
    else:
        ckpt["procgen_state"] = None
        ckpt["envs_pickle"] = _try_pickle(envs, "vectorized envs")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


def restore_checkpoint(
    ckpt,
    *,
    args,
    device,
    agent,
    r_in_fn,
    r_rnd_fn,
    optim,
    r_in_optimizer,
    r_rnd_optimizer,
    rb_ppg,
    envs,
    r_in_target_fn=None,
):
    """Restore models and optimizers in place; return the buffers, envs and counters train() reassigns."""
    agent.load_state_dict(ckpt["agent_state_dict"])
    r_in_fn.load_state_dict(ckpt["r_in_fn_state_dict"])
    if r_in_target_fn is not None:
        target_sd = ckpt.get("r_in_target_fn_state_dict")
        if target_sd is not None:
            r_in_target_fn.load_state_dict(target_sd)
        else:
            r_in_target_fn.load_state_dict(ckpt["r_in_fn_state_dict"])
    r_rnd_fn.load_state_dict(ckpt["r_rnd_fn_state_dict"])
    rnd_extras = ckpt["r_rnd_extras"]
    r_rnd_fn.obs_rms.mean = rnd_extras["obs_rms_mean"]
    r_rnd_fn.obs_rms.var = rnd_extras["obs_rms_var"]
    r_rnd_fn.obs_rms.count = rnd_extras["obs_rms_count"]
    r_rnd_fn.rew_rms.mean = rnd_extras["rew_rms_mean"]
    r_rnd_fn.rew_rms.var = rnd_extras["rew_rms_var"]
    r_rnd_fn.rew_rms.count = rnd_extras["rew_rms_count"]
    r_rnd_fn.discounted_reward.rewems = rnd_extras["discounted_rewems"]

    optim.load_state_dict(ckpt["optim_state_dict"])
    r_in_optimizer.load_state_dict(ckpt["r_in_optimizer_state_dict"])
    r_rnd_optimizer.load_state_dict(ckpt["r_rnd_optimizer_state_dict"])

    if rb_ppg is not None and ckpt["rb_ppg_pickle"] is not None:
        rb_ppg = pickle.loads(ckpt["rb_ppg_pickle"])

    procgen_state = ckpt.get("procgen_state")
    if procgen_state is not None:
        assert isinstance(envs, ProcGenVecEnv), (
            "Checkpoint contains procgen_state but current envs are not "
            "ProcGenVecEnv; were run args changed?"
        )
        envs.set_state(procgen_state)
    elif ckpt["envs_pickle"] is not None:
        try:
            envs.close()
        except Exception:
            pass
        envs = pickle.loads(ckpt["envs_pickle"])
    else:
        print(
            "Warning: checkpoint has no pickled envs; trajectories will diverge "
            "from an uninterrupted run after resume.",
            flush=True,
        )

    args.freeze_rin_frac = ckpt["mutable_args"]["freeze_rin_frac"]
    args.meta_update = ckpt["mutable_args"]["meta_update"]

    _restore_rng_state(ckpt["rng_state"])
    start_iteration = ckpt["iteration"] + 1
    print(
        f"Resumed from checkpoint at iteration {ckpt['iteration']} "
        f"(global_step={ckpt['global_step']}); continuing at iteration {start_iteration}.",
        flush=True,
    )

    return {
        "rb_ppg": rb_ppg,
        "envs": envs,
        "next_obs": ckpt["next_obs"].to(device),
        "next_done": ckpt["next_done"].to(device),
        "global_step": ckpt["global_step"],
        "match_step": ckpt["match_step"],
        "return_timesteps": ckpt["return_timesteps"],
        "delay_r_ex": ckpt["delay_r_ex"],
        "delay_step": ckpt["delay_step"],
        "start_iteration": start_iteration,
        "start_time": time.time() - ckpt["elapsed_time"],
        "practice_state": ckpt.get("practice_state"),
        "adam_snr_state": ckpt.get("adam_snr_state"),
        "gradnorm_state": ckpt.get("gradnorm_state"),
    }
