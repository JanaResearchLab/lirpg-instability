"""Figures 12-14: grid of smoothed learning curves, one subplot per game.

Usage: python plotting/plot_learning_curve_grid.py --runs_dir runs-atari-complete [--aggregate] [--games 0:28]
"""

import argparse
import math
import os
import re
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from common_utils import (
    MultiRunDataCache,
    baseline_event_paths,
    get_event_data_from_event_file,
    load_baseline_paths,
    organize_paths_by_method,
)
from tqdm import tqdm

# Exact names from common_utils.map_method_name.
_PPO_LIRPG_NAME = "PPO-LIRPG ($r^\\mathrm{in}$)"
_DCPG_NAME = "PPO-LIRPG ($r^\\mathrm{in}$) (DCPG)"

METHOD_CONFIG = {
    "PPO-LIRPG": {
        "exact_name": _PPO_LIRPG_NAME,
        "label": r"PPO-LIRPG ($r^{\mathrm{in}}$)",
        "color": "#e6850e",
    },
    "DCPG": {
        "exact_name": _DCPG_NAME,
        "label": r"PPO-LIRPG ($r^{\mathrm{in}}$) (DCPG)",
        "color": "#2ca02c",
    },
}

_NAME_TO_KEY = {cfg["exact_name"]: k for k, cfg in METHOD_CONFIG.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a grid of learning curves, one subplot per game."
    )
    parser.add_argument("--runs_dir", type=str, default="runs-atari-complete")
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate seeds with mean +/- std instead of plotting each seed.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.99,
        help="EMA smoothing factor in [0, 1). Higher = smoother.",
    )
    parser.add_argument(
        "--n_cols",
        type=int,
        default=5,
        help="Number of subplot columns in the grid.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="charts/episodic_return",
        help="TF event tag to plot.",
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--games",
        type=str,
        default=None,
        help="start:end slice of the alphabetical game list (Figure 12 is 0:28, Figure 13 is 28:).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path. If None, saves under runs_dir/figures/.",
    )
    return parser.parse_args()


def match_method(mapped_name):
    return _NAME_TO_KEY.get(mapped_name, mapped_name)


def detect_games(runs_dir):
    games = set()
    for folder in os.listdir(runs_dir):
        full_path = os.path.join(runs_dir, folder)
        if not os.path.isdir(full_path):
            continue
        m = re.match(r"(.+?)__--", folder)
        if m:
            games.add(m.group(1))
    return sorted(games)


PROCGEN_NAMES = {
    "caveflyer": "CaveFlyer",
    "coinrun": "CoinRun",
    "dodgeball": "Dodgeball",
    "fruitbot": "FruitBot",
    "starpilot": "StarPilot",
}


def prettify_game_name(game):
    if game.startswith("procgen-"):
        return PROCGEN_NAMES.get(game[len("procgen-") :], game)
    name = game.replace("NoFrameskip-v4", "")
    name = re.sub(r"-v\d+$", "", name)
    return name


def load_curve(path, cache, metric):
    """Return (steps, values) for one event file, going through the cache."""
    if cache is not None:
        cached = cache.get_cached_data(path, metric)
        if cached is not None:
            steps, values, _ = cached
            return np.asarray(steps, dtype=float), np.asarray(values, dtype=float)

    events = get_event_data_from_event_file(path, metric)
    if events is None:
        return None, None
    steps = [e.step for e in events]
    values = [e.value for e in events]
    if cache is not None:
        cache.cache_data(
            path, steps, values, list(range(1, len(values) + 1)), metric=metric
        )
    return np.asarray(steps, dtype=float), np.asarray(values, dtype=float)


def exponential_moving_average(data, smoothing_factor=0.9):
    """TensorBoard-style EMA. Identity for empty input."""
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return data
    alpha = 1.0 - smoothing_factor
    smoothed = np.empty_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def collect_curves(runs_dir, metric, cache):
    """Walk runs_dir and return {game: {method_key: [(steps, values), ...]}}."""
    baseline_folders = load_baseline_paths(runs_dir)

    games = detect_games(runs_dir)
    games = sorted(
        set(games) | {os.path.basename(f).split("__--")[0] for f in baseline_folders}
    )
    print(f"Detected {len(games)} games.")

    curves = defaultdict(lambda: defaultdict(list))

    for game in tqdm(games, desc="Games"):
        folder_pattern = f"{game}__--*--*"
        game_paths = sorted(
            glob(os.path.join(runs_dir, folder_pattern, "events.out.tfevents.*"))
        )
        game_paths += baseline_event_paths(baseline_folders, folder_pattern)
        game_paths = sorted(set(game_paths))
        print(f"  Found {len(game_paths)} event files for game {game}.")
        if not game_paths:
            continue

        methods_paths = organize_paths_by_method(game_paths)
        for method_name, paths in methods_paths.items():
            config_key = match_method(method_name)
            if config_key not in METHOD_CONFIG:
                continue
            for path in paths:
                steps, values = load_curve(path, cache, metric)
                if steps is None or len(steps) < 2:
                    continue
                curves[game][config_key].append((steps, values))

    return curves


def interpolate_to_common(steps_list, values_list, num_points=500):
    """Interpolate seeds onto a common step grid bounded by their overlap."""
    valid = [(s, v) for s, v in zip(steps_list, values_list) if len(s) >= 2]
    if not valid:
        return np.array([]), np.array([])
    s_min = max(s[0] for s, _ in valid)
    s_max = min(s[-1] for s, _ in valid)
    if s_max <= s_min:
        return np.array([]), np.array([])
    common = np.linspace(s_min, s_max, num_points)
    interp = np.array([np.interp(common, s, v) for s, v in valid])
    return common, interp


def build_method_display(curves):
    """Return (ordered keys, {key: {label, color}}) for the methods present, in METHOD_CONFIG order."""
    seen = {k for game_curves in curves.values() for k in game_curves}
    ordered = [k for k in METHOD_CONFIG if k in seen]
    return ordered, {
        k: {"label": METHOD_CONFIG[k]["label"], "color": METHOD_CONFIG[k]["color"]}
        for k in ordered
    }


def plot_grid(curves, aggregate, smoothing, n_cols, output_path):
    games = sorted(curves.keys(), key=prettify_game_name)
    n = len(games)
    if n == 0:
        print("No curves to plot.")
        return

    method_order, method_display = build_method_display(curves)

    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.6 * n_cols, 2.0 * n_rows),
        squeeze=False,
    )

    used_keys = set()
    for i, game in enumerate(games):
        ax = axes[i // n_cols][i % n_cols]
        method_curves = curves[game]

        for method_key in method_order:
            cfg = method_display[method_key]
            seeds = method_curves.get(method_key, [])
            if not seeds:
                continue
            used_keys.add(method_key)

            steps_list = [s for s, _ in seeds]
            values_list = [v for _, v in seeds]

            if aggregate:
                common, interp = interpolate_to_common(steps_list, values_list)
                if len(common) == 0:
                    continue
                smoothed = np.array(
                    [exponential_moving_average(v, smoothing) for v in interp]
                )
                mean = smoothed.mean(axis=0)
                std = smoothed.std(axis=0)
                ax.plot(common, mean, color=cfg["color"], linewidth=1.4)
                ax.fill_between(
                    common,
                    mean - std,
                    mean + std,
                    color=cfg["color"],
                    alpha=0.2,
                    linewidth=0,
                )
            else:
                for s, v in zip(steps_list, values_list):
                    smoothed = exponential_moving_average(v, smoothing)
                    ax.plot(s, smoothed, color=cfg["color"], linewidth=0.8)

        ax.set_title(prettify_game_name(game), fontsize=9)
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0)

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    legend_keys = [k for k in method_order if k in used_keys]
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=method_display[k]["color"],
            label=method_display[k]["label"],
            linewidth=2,
        )
        for k in legend_keys
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="lower right",
            bbox_to_anchor=(0.995, 0.005),
            fontsize=9,
            framealpha=0.9,
        )

    fig.supxlabel("Steps", fontsize=10)
    fig.supylabel("Episodic Return", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path} and {pdf_path}")


def main():
    args = parse_args()

    cache = MultiRunDataCache(args.runs_dir, primary_cache_dir=args.cache_dir)

    curves = collect_curves(args.runs_dir, args.metric, cache)
    cache.save()
    if args.games:
        start, end = (int(x) if x else None for x in args.games.split(":"))
        keep = sorted(curves, key=prettify_game_name)[start:end]
        curves = {g: curves[g] for g in keep}

    figures_dir = os.path.join(args.runs_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    suffix = "agg" if args.aggregate else "seeds"
    part = f"_{args.games.replace(':', '-')}" if args.games else ""
    output_path = args.output or os.path.join(
        figures_dir, f"learning_curve_grid_{suffix}{part}.png"
    )

    plot_grid(
        curves,
        aggregate=args.aggregate,
        smoothing=args.smoothing,
        n_cols=args.n_cols,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
