"""Figure 9: aggregate normalized score of the alternate strategies on the 7 Atari games.

Usage: python plotting/plot_alt_strategies_figure.py
"""

import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from common_utils import (
    MultiRunDataCache,
    get_random_policy_scores,
    normalize_method_scores,
)
from compare_methods_rliable import plot_interval_estimates_vertical
from rliable import library as rly
from rliable import metrics

RUNS_DIR = "runs-atari-alternate-strategies"
OUTPUT_DIR = os.path.join(RUNS_DIR, "figures-rliable")

ENVS = [
    f"{g}NoFrameskip-v4"
    for g in (
        "BeamRider",
        "Breakout",
        "Enduro",
        "Pong",
        "Qbert",
        "Seaquest",
        "SpaceInvaders",
    )
]

BASELINE = "PPO-LIRPG ($r^\\mathrm{in}$)"
DCPG = "PPO-LIRPG ($r^\\mathrm{in}$) (DCPG)"


def _strategy(tag):
    return {env: f"{RUNS_DIR}/{env}__--tuning_0_1_0-{tag}--__*__*" for env in ENVS}


# label -> {env: run folder glob}
METHOD_PATTERNS = {
    BASELINE: {env: f"runs-atari-tuning/{env}__--tuning_0_1_0--__*__*" for env in ENVS},
    DCPG: {
        env: f"runs-atari-dcpg-reg/{env}__--tuning_0_1_0-ppg_vr_0.0_pr_1.0_tfreq_32--__*__*"
        for env in ENVS
    },
    "PPO-LIRPG ($r^\\mathrm{in}$) ($r^\\mathrm{in}$@$r^\\mathrm{ex}$)": _strategy(
        "r_in_if_r_ex"
    ),
    "PPO-LIRPG ($r^\\mathrm{in}$) (scaled $r^\\mathrm{in}$)": _strategy(
        "scale_rin_dens"
    ),
    "PPO-LIRPG ($r^\\mathrm{in}$) (scaled value)": _strategy("vcoef_0.05"),
    "PPO-LIRPG ($r^\\mathrm{in}$) (reduced $\\gamma=0.9$)": _strategy("gamma_0.9"),
    "PPO-LIRPG ($r^\\mathrm{in}$) (PopArt)": _strategy("popart"),
    "PPO-LIRPG ($r^\\mathrm{in}$) (target $r^\\mathrm{in}$)": _strategy(
        "target_rin_tau0.02"
    ),
    "PPO-LIRPG ($r^\\mathrm{in}$) (PCGrad)": _strategy("pcgrad"),
    "PPO-LIRPG ($r^\\mathrm{in}$) (GradNorm)": _strategy("gradnorm"),
}

BAR_ORDER = list(METHOD_PATTERNS)
COLORS = {
    BASELINE: "#2ca02c",
    DCPG: "#80ff80",
    "PPO-LIRPG ($r^\\mathrm{in}$) ($r^\\mathrm{in}$@$r^\\mathrm{ex}$)": "#9467bd",
    "PPO-LIRPG ($r^\\mathrm{in}$) (scaled $r^\\mathrm{in}$)": "#7f7f7f",
    "PPO-LIRPG ($r^\\mathrm{in}$) (scaled value)": "#17becf",
    "PPO-LIRPG ($r^\\mathrm{in}$) (reduced $\\gamma=0.9$)": "#e377c2",
    "PPO-LIRPG ($r^\\mathrm{in}$) (PopArt)": "#8c564b",
    "PPO-LIRPG ($r^\\mathrm{in}$) (target $r^\\mathrm{in}$)": "#bcbd22",
    "PPO-LIRPG ($r^\\mathrm{in}$) (PCGrad)": "#d62728",
    "PPO-LIRPG ($r^\\mathrm{in}$) (GradNorm)": "#1a9988",
}
HATCHES = ["//" if m == DCPG else "" for m in BAR_ORDER]


def resolve_event_paths(pattern):
    dirs = sorted(glob(pattern))
    paths = []
    for d in dirs:
        paths += sorted(glob(os.path.join(d, "events.out.tfevents.*")))
    if not paths:
        raise ValueError(f"No runs matched pattern: {pattern}")
    return paths


def build_game_method_paths():
    game_method_paths = {env: {} for env in ENVS}
    for method, env_patterns in METHOD_PATTERNS.items():
        for env, pattern in env_patterns.items():
            game_method_paths[env][method] = resolve_event_paths(pattern)
    return game_method_paths


def scores_matrix(normalized_scores, method, envs):
    """(num_seeds, num_envs) array of normalized scores for one method."""
    per_env = [normalized_scores[env][method] for env in envs]
    min_seeds = min(len(s) for s in per_env)
    return np.array([s[:min_seeds] for s in per_env]).T


def aggregate_func(x):
    return np.array([metrics.aggregate_mean(x)])


def plot_for_window(game_method_paths, cache, window):
    random_policy_scores = get_random_policy_scores(
        RUNS_DIR,
        use_median=True,
        window=window,
        cache=cache,
        baseline_folders=sorted(glob("runs-atari-tuning/*__--random_policy--__*")),
    )
    normalized_scores, _ = normalize_method_scores(
        game_method_paths,
        random_policy_scores,
        map_names=False,
        cache=cache,
        window=window,
    )
    cache.save()

    matrices = {m: scores_matrix(normalized_scores, m, ENVS) for m in BAR_ORDER}
    point_estimates, interval_estimates = rly.get_interval_estimates(
        matrices, aggregate_func, reps=50000
    )

    print(f"\n=== window={window} ===")
    for method in BAR_ORDER:
        lo, hi = interval_estimates[method][:, 0]
        print(f"  {method}: {point_estimates[method][0]:.3f} [{lo:.3f}, {hi:.3f}]")

    fig, axes = plot_interval_estimates_vertical(
        point_estimates,
        interval_estimates,
        metric_names=["Mean"],
        algorithms=BAR_ORDER,
        colors=COLORS,
        hatches=HATCHES,
        legend_location="right",
    )
    ax_min = min(interval_estimates[m][0, 0] for m in BAR_ORDER)
    for ax in axes:
        ax.set_ylim(min(ax_min, 0), 1.1)
    fig_path = os.path.join(OUTPUT_DIR, f"aggregate_metrics-w={window}.png")
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved aggregate metrics plot to {fig_path}")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    game_method_paths = build_game_method_paths()
    cache = MultiRunDataCache(RUNS_DIR)
    for window in (100, 0):
        plot_for_window(game_method_paths, cache, window)


if __name__ == "__main__":
    main()
