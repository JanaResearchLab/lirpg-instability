"""Print Tables 4-5 and 8-13 of the paper from TensorBoard logs.

Usage: python plotting/paper_tables.py {table4,table5,table8,...,table13} [--runs_dir DIR] [--exp EXP]
"""

import argparse
import json
import os
import re
from glob import glob

import numpy as np
from common_utils import (
    MultiRunDataCache,
    get_random_policy_scores,
    normalize_method_scores,
)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ATARI7 = [
    "BeamRider",
    "Breakout",
    "Enduro",
    "Pong",
    "Qbert",
    "Seaquest",
    "SpaceInvaders",
]
ATARI5 = ["BeamRider", "Breakout", "Pong", "Seaquest", "SpaceInvaders"]
EXCEPTION_SEEDS = {
    "DemonAttack": [0, 1, 2],
    "KungFuMaster": [0, 1, 2],
    "Centipede": [0, 1, 2],
    "Asteroids": [0, 1, 2],
    "Pitfall": [0, 1, 2],
    "IceHockey": [0, 1, 2],
}
RANDOM_POLICY_GLOB = "runs-atari-tuning/*__--random_policy--__*"


def env_id(game):
    return f"{game}NoFrameskip-v4"


def run_dirs(runs_dir, game, exp):
    return sorted(glob(os.path.join(runs_dir, f"{env_id(game)}__--{exp}--__*__*")))


def scalars(run_dir, tags):
    """{tag: np.array of per-iteration values} for the requested tags."""
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags()["scalars"])
    return {t: np.array([e.value for e in ea.Scalars(t)]) for t in tags if t in have}


def pooled(runs, tag):
    """Per-iteration values of `tag`, pooled across the runs (seeds)."""
    return np.concatenate([scalars(r, [tag]).get(tag, np.array([])) for r in runs])


def fmt(x, digits=3):
    return f"{x:.{digits}g}" if abs(x) >= 1 else f"{x:.{digits}f}"


# ------------------------------------------------------------------ Tables 4-5
def table4(args):
    for runs_dir, postfix in (
        (args.runs_dir, ""),
        (args.dcpg_runs_dir, args.dcpg_postfix),
    ):
        if runs_dir is None:
            continue
        path = os.path.join(runs_dir, "best_runs.json")
        if not os.path.exists(path):
            print(
                f"{path} missing: run plotting/visualize_hyperparameters.py on {runs_dir} first"
            )
            continue
        best = json.load(open(path))["PPO-LIRPG"]
        best = best.get(postfix, best)
        print(f"\n{runs_dir} (postfix '{postfix}'): chosen lambda_in")
        for env, paths in sorted(best.items()):
            rin = sorted(
                {re.search(r"tuning_[^_]+_([^_]+)_", p).group(1) for p in paths}
            )
            print(f"  {env.replace('NoFrameskip-v4', ''):28s} {', '.join(rin)}")


table5 = table4


# ------------------------------------------------------------------ Table 8 / 13
GRAD_TAGS = {
    "||g_V|| (scaled)": "losses/inner_grad_v_scaled",
    "||g_pi||": "losses/inner_grad_p",
    "alpha_V": "losses/inner_grad_alpha_est",
    "rho": "losses/inner_grad_dominance_ratio",
    "cos(g_V,g_pi)": "losses/inner_grad_pv_cosine_encoder",
    "cos(g,g_pi)": "losses/inner_grad_p_similarity",
    "cos(g,g_V)": "losses/inner_grad_v_similarity",
}


MEDIAN_TAGS = ("alpha_V", "rho", "cos(g,g_pi)", "cos(g,g_V)")


def grad_row(runs):
    """Mean over iterations and seeds, or median of per-run medians for MEDIAN_TAGS."""
    row = {}
    for name, tag in GRAD_TAGS.items():
        per_run = [
            v
            for v in (scalars(r, [tag]).get(tag, np.array([])) for r in runs)
            if v.size
        ]
        if not per_run:
            row[name] = float("nan")
        elif name in MEDIAN_TAGS:
            row[name] = float(np.median([np.median(v) for v in per_run]))
        else:
            row[name] = float(np.mean(np.concatenate(per_run)))
    return row


def table8(args):
    names = list(GRAD_TAGS)
    print("env            " + "  ".join(f"{n:>16s}" for n in names))
    rows = {}
    for g in ATARI7:
        runs = run_dirs(args.runs_dir, g, args.exp)
        rows[g] = grad_row(runs)
        print(
            f"{g:14s} "
            + "  ".join(f"{fmt(rows[g][n]):>16s}" for n in names)
            + f"   (n={len(runs)})"
        )
    ext = {g: grad_row(run_dirs(args.runs_dir, g, args.ext_exp)) for g in ATARI7}
    for label, rs in (("Aggregate r_in", rows), ("Aggregate r_ex", ext)):
        agg = {
            n: (np.median if n in ("alpha_V", "rho") else np.mean)(
                [rs[g][n] for g in ATARI7]
            )
            for n in names
        }
        rho = [rs[g]["rho"] for g in ATARI7]
        print(
            f"{label:14s} "
            + "  ".join(f"{fmt(agg[n]):>16s}" for n in names)
            + f"   (rho IQR {fmt(np.percentile(rho, 25, method='lower'))}-{fmt(np.percentile(rho, 75, method='higher'))})"
        )


def table13(args):
    """Every run cut at --cut steps; medians over iterations."""
    print(
        f"{'game':14s} {'seed':>4s} {'return':>10s} {'||g_pi||':>10s} {'rho':>10s} {'cos(g,g_pi)':>12s}"
    )
    for g, seeds in EXCEPTION_SEEDS.items():
        for seed in seeds:
            (r,) = glob(
                os.path.join(args.runs_dir, f"{env_id(g)}__--{args.exp}--__{seed}__*")
            )
            ea = EventAccumulator(r, size_guidance={"scalars": 0})
            ea.Reload()
            cut = lambda tag: np.array(
                [e.value for e in ea.Scalars(tag) if e.step <= args.cut]
            )
            med = {
                n: np.median(cut(GRAD_TAGS[n]))
                for n in ("||g_pi||", "rho", "cos(g,g_pi)")
            }
            ret = np.mean(cut("charts/episodic_return")[-100:])
            print(
                f"{g:14s} {seed:4d} {ret:10.0f} {med['||g_pi||']:10.2g} {med['rho']:10.2g} "
                f"{med['cos(g,g_pi)']:12.3f}"
            )


# ------------------------------------------------------------------ Table 9
def table9(args):
    tags = {
        ("Policy phase", "KL"): "losses/inner_phasic_reg/policy_phase_policy_drift",
        ("Policy phase", "MSE"): "losses/inner_phasic_reg/policy_phase_value_drift",
        ("Value phase", "KL"): "losses/inner_phasic_reg/value_phase_policy_drift",
        ("Value phase", "MSE"): "losses/inner_phasic_reg/value_phase_value_drift",
    }
    runs = [r for g in ATARI7 for r in run_dirs(args.runs_dir, g, args.exp)]
    print(
        f"{len(runs)} runs; per iteration, so value-phase changes are divided by t_freq={args.t_freq}; "
        "value change is logged as 0.5 * MSE and doubled here; median over iterations, mean over runs"
    )
    print(f"{'Phase':14s} {'Policy change (KL)':>20s} {'Value change (MSE)':>20s}")
    for phase, div in (("Policy phase", 1), ("Value phase", args.t_freq)):

        def run_mean(r, tag):
            v = scalars(r, [tag])[tag]
            return np.median(v[np.isfinite(v)]) / div

        kl = np.mean([run_mean(r, tags[(phase, "KL")]) for r in runs])
        mse = 2 * np.mean([run_mean(r, tags[(phase, "MSE")]) for r in runs])
        print(f"{phase:14s} {kl:20.6f} {mse:20.4f}")


# ------------------------------------------------------------------ Tables 10-12
def normalized_table(methods, games, runs_dir, window=100):
    """Print mean +- s.d. over seeds of the mean normalized score; methods: {label: (runs_dir, exp)}."""
    game_method_paths = {}
    for g in games:
        game_method_paths[env_id(g)] = {}
        for label, (d, exp) in methods.items():
            paths = [
                p
                for r in run_dirs(d, g, exp)
                for p in sorted(glob(os.path.join(r, "events.out.tfevents.*")))
            ]
            if not paths:
                raise SystemExit(f"no runs for {label} on {g} ({d}, {exp})")
            game_method_paths[env_id(g)][label] = paths
    cache = MultiRunDataCache(runs_dir)
    random_scores = get_random_policy_scores(
        runs_dir,
        use_median=True,
        window=window,
        cache=cache,
        baseline_folders=sorted(glob(RANDOM_POLICY_GLOB)),
    )
    normalized, _ = normalize_method_scores(
        game_method_paths, random_scores, map_names=False, cache=cache, window=window
    )
    cache.save()
    for label in methods:
        per_game = [normalized[env_id(g)][label] for g in games]
        n = min(len(s) for s in per_game)
        per_seed = np.mean([s[:n] for s in per_game], axis=0)
        print(
            f"  {label:40s} {per_seed.mean():.3f} +- {per_seed.std(ddof=1):.3f}   (seeds={n})"
        )


def table10(args):
    d = args.runs_dir
    normalized_table(
        {
            "Shared encoder": (args.baseline_runs_dir, "tuning_0_1_0"),
            "Separate encoder (policy-pretrained)": (d, "pretrained_policy_only"),
            "Shared encoder + DCPG": (args.dcpg_runs_dir, args.dcpg_exp),
        },
        ATARI7,
        d,
    )


def table11(args):
    methods = {"PPO-LIRPG(r_in)": (args.baseline_runs_dir, "tuning_0_1_0")}
    for t in (1, 8, 16, 32, 64):
        d = args.dcpg_runs_dir if t == 32 else args.runs_dir
        methods[f"PPO-LIRPG(r_in) (DCPG) t_freq={t}"] = (
            d,
            f"tuning_0_1_0-ppg_vr_0.0_pr_1.0_tfreq_{t}",
        )
    normalized_table(methods, ATARI5, args.runs_dir)


def table12(args):
    methods = {
        f"lambda_V={vf}": (args.runs_dir, f"tuning_0_1_0-vf_{vf}")
        for vf in ("0.01", "0.05", "0.1", "0.25", "0.5")
    }
    normalized_table(methods, ATARI5, args.runs_dir)


DEFAULTS = {
    "table4": dict(
        runs_dir="runs-atari-tuning",
        dcpg_runs_dir="runs-atari-tuning-dcpg",
        dcpg_postfix="-dcpg",
    ),
    "table5": dict(runs_dir="runs-mujoco-tuning"),
    "table8": dict(
        runs_dir="runs-atari-diagnostic-grad_stats_compare",
        exp="tuning_0_1_0-grad_stats",
        ext_exp="tuning_1_0_0-grad_stats_ppo",
    ),
    "table9": dict(
        runs_dir="runs-atari-dcpg-phase-drift",
        exp="tuning_0_1_0-dcpg_phasic",
        t_freq=32,
    ),
    "table10": dict(
        runs_dir="runs-atari-comparison-pretrained_init",
        baseline_runs_dir="runs-atari-tuning",
        dcpg_runs_dir="runs-atari-dcpg-reg",
        dcpg_exp="tuning_0_1_0-ppg_vr_0.0_pr_1.0_tfreq_32",
    ),
    "table11": dict(
        runs_dir="runs-atari-dcpg-tfreq",
        baseline_runs_dir="runs-atari-tuning",
        dcpg_runs_dir="runs-atari-dcpg-reg",
    ),
    "table12": dict(runs_dir="runs-atari-vf_coef_sweep"),
    "table13": dict(
        runs_dir="runs-atari-diagnostic-grad_stats",
        exp="tuning_0_1_0-grad_stats",
        cut=5_000_000,
    ),
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("table", choices=sorted(DEFAULTS))
    for key in (
        "runs_dir",
        "exp",
        "ext_exp",
        "dcpg_runs_dir",
        "dcpg_exp",
        "dcpg_postfix",
        "baseline_runs_dir",
    ):
        parser.add_argument(f"--{key}")
    parser.add_argument("--cut", type=int)
    parser.add_argument("--t_freq", type=int)
    args = parser.parse_args()
    for k, v in DEFAULTS[args.table].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    globals()[args.table](args)


if __name__ == "__main__":
    main()
