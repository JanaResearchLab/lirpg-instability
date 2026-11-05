import argparse
import os
import subprocess
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from common_utils import (
    MultiRunDataCache,
    RunDataCache,
    baseline_event_paths,
    get_all_postfixes,
    get_event_data_from_event_file,
    get_random_policy_scores,
    load_baseline_paths,
    method_name_to_setting,
    normalize_method_scores,
    organize_paths_by_method,
)
from rliable import library as rly
from rliable import metrics, plot_utils
from scipy.stats import combine_pvalues, mannwhitneyu


def plot_interval_estimates_vertical(
    point_estimates,
    interval_estimates,
    metric_names,
    algorithms=None,
    colors=None,
    color_palette="colorblind",
    hatches=None,
    max_ticks=6,
    subfigure_width=3.4,
    xlabel="Normalized Score",
    legend_location=None,
    **kwargs,
):
    """Vertical-bar version of rliable's plot_utils.plot_interval_estimates, with a legend."""
    if algorithms is None:
        algorithms = list(point_estimates.keys())
    num_metrics = len(point_estimates[algorithms[0]])
    figsize = (max(subfigure_width * num_metrics, 0.5 * len(algorithms)), 4)
    fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=figsize)
    axes = np.atleast_1d(axes)
    if colors is None:
        color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
        colors = dict(zip(algorithms, color_palette))
    if hatches is None:
        hatches = [""] * len(algorithms)
    h = kwargs.pop("interval_height", 0.6)

    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        for alg_idx, algorithm in enumerate(algorithms):
            lower, upper = interval_estimates[algorithm][:, idx]
            ax.bar(
                x=alg_idx,
                height=upper - lower,
                width=h,
                bottom=lower,
                color=colors[algorithm],
                alpha=0.75,
                label=algorithm,
                hatch=hatches[alg_idx] or None,
                edgecolor="black" if hatches[alg_idx] else None,
            )
            ax.hlines(
                y=point_estimates[algorithm][idx],
                xmin=alg_idx - (7.5 * h / 16),
                xmax=alg_idx + (6 * h / 16),
                color="k",
                alpha=0.5,
            )
        ax.set_xticks([])
        ax.set_xlim(-0.75, len(algorithms) - 0.25)
        ax.yaxis.set_major_locator(plt.MaxNLocator(max_ticks))
        if len(metric_names) > 1:
            ax.set_title(metric_name, fontsize="xx-large")
        ax.tick_params(axis="y", which="major", labelsize="xx-large")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel(xlabel, fontsize="xx-large")
    if legend_location:
        loc = "center left" if legend_location == "right" else legend_location
        bbox = (1.02, 0.5) if legend_location == "right" else None
        axes[-1].legend(loc=loc, bbox_to_anchor=bbox, fontsize="x-large", frameon=False)
    plt.subplots_adjust(wspace=kwargs.pop("wspace", 0.3))
    return fig, list(axes)


def get_all_frames_scores_dict(
    game_method_paths,
    cache: RunDataCache | None = None,
    tag: str = "charts/episodic_return",
):
    ale_all_frames_scores_dict = {}
    for _, methods_paths in game_method_paths.items():
        for method, paths in methods_paths.items():
            ale_all_frames_scores_dict[method] = ale_all_frames_scores_dict.get(
                method, []
            )
            game_scores = []
            for path in paths:
                if cache and cache._is_cache_valid(
                    path, cache.get_cache_key(path, tag)
                ):
                    steps, returns, _ = cache.get_cached_data(path, tag)
                else:
                    events = get_event_data_from_event_file(path, tag)
                    returns = [e.value for e in events]
                    steps = [e.step for e in events]
                selected_returns = []
                # One return per 100k env steps.
                for i, step in enumerate(steps):
                    if 100000 * len(selected_returns) - step <= 0:
                        selected_returns.append(returns[i])
                game_scores.append(np.array(selected_returns))
            ale_all_frames_scores_dict[method].append(np.array(game_scores))

    scores_dict = {}
    for method, scores in ale_all_frames_scores_dict.items():
        scores_dict[method] = np.array(scores)
        scores_dict[method] = np.transpose(scores_dict[method], (1, 0, 2))

    return scores_dict


def run_compare_methods(args):
    """Run compare_methods.py on the same runs and filters in a subprocess."""
    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "compare_methods.py"
    )
    cmd = [sys.executable, script, "--runs_dir", args.runs_dir]
    # compare_methods.py takes comma-separated strings where this script takes lists.
    if args.postfixes:
        cmd += ["--postfixes", ",".join(args.postfixes)]
    if args.settings:
        cmd += ["--settings", ",".join(args.settings)]
    if args.skip_methods:
        cmd += ["--skip_methods", ",".join(args.skip_methods)]
    if args.only_methods:
        cmd += ["--only_methods", ",".join(args.only_methods)]
    if args.coupled:
        cmd.append("--coupled")

    print("\nRunning:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"compare_methods.py failed with exit code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="runs",
        help="Directory with tensorboard event folders",
    )
    parser.add_argument(
        "--output_dir", type=str, default="", help="Directory to save output figures"
    )
    parser.add_argument(
        "--postfixes",
        type=str,
        default=None,
        help="Comma-separated run postfixes to include, or 'all'.",
    )
    parser.add_argument(
        "--settings",
        type=str,
        nargs="+",
        default=None,
        help="Settings to include (default: all).",
    )
    parser.add_argument(
        "--coupled",
        action="store_true",
        help="Pair adjacent methods by color and hatch.",
    )
    parser.add_argument(
        "--skip_methods",
        type=str,
        nargs="+",
        default=None,
        help="Methods to skip (fnmatch wildcards; aliases mix, in, ex).",
    )
    parser.add_argument(
        "--skip_games",
        type=str,
        nargs="+",
        default=None,
        help="Games to skip.",
    )
    parser.add_argument(
        "--only_methods",
        type=str,
        nargs="+",
        default=None,
        help="Methods to keep (fnmatch wildcards; aliases mix, in, ex).",
    )
    parser.add_argument(
        "--common_envs",
        action="store_true",
        help="Aggregate over games that have every method, not methods common to all games.",
    )
    parser.add_argument(
        "--vertical",
        action="store_true",
        help="Plot vertical interval estimates.",
    )
    parser.add_argument(
        "--n_plots",
        type=int,
        default=1,
        help="How many plots to make, in order: aggregate, improvement, efficiency, profiles.",
    )
    parser.add_argument(
        "--plot_all",
        action="store_true",
        help="Also run compare_methods.py with the same filters.",
    )
    parser.add_argument(
        "--color_offset",
        type=int,
        default=0,
        help="Index of the first palette color.",
    )
    args = parser.parse_args()
    args.postfixes = args.postfixes.split(",") if args.postfixes is not None else None

    if args.output_dir == "":
        postfixes_suffix = ",".join(args.postfixes) if args.postfixes else ""
        args.output_dir = os.path.join(
            args.runs_dir,
            f"figures-rliable{postfixes_suffix}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    if args.postfixes == ["all"]:
        postfixes = sorted(get_all_postfixes(args.runs_dir))
    else:
        postfixes = args.postfixes if args.postfixes else []
    settings = args.settings if args.settings else []

    domain = (
        "Atari"
        if "atari" in args.runs_dir
        else "MuJoCo"
        if "mujoco" in args.runs_dir
        else "dmcontrol"
    )

    baseline_folders = load_baseline_paths(args.runs_dir)

    all_dirs = glob(os.path.join(args.runs_dir, "*--*--*"))
    games = set([os.path.basename(path).split("__--")[0] for path in all_dirs])
    games |= {os.path.basename(f).split("__--")[0] for f in baseline_folders}
    games = sorted(games)
    if args.skip_games is not None:
        skip_games = set(args.skip_games)
        games = [game for game in games if game not in skip_games]
        print(f"Skipped games: {skip_games}")
    game_method_paths = {}
    for game in games:
        game_paths = []
        if args.postfixes is None:
            folder_pattern = f"{game}*--*--*"
            path_pattern = os.path.join(
                args.runs_dir, folder_pattern, "events.out.tfevents.*"
            )
            game_paths = sorted(glob(path_pattern))
            game_paths += baseline_event_paths(baseline_folders, folder_pattern)
        else:
            for postfix in postfixes:
                folder_pattern = f"{game}*--*_*_0{postfix}--*"  # r_rnd = 0 runs only
                path_pattern = os.path.join(
                    args.runs_dir,
                    folder_pattern,
                    "events.out.tfevents.*",
                )
                paths = sorted(glob(path_pattern))
                # Empty postfix must exclude runs that carry any postfix.
                pred = (
                    (lambda ev: "-" not in ev.split("--")[1]) if postfix == "" else None
                )
                if postfix == "":
                    paths = [p for p in paths if pred(p)]
                game_paths += paths
                game_paths += baseline_event_paths(
                    baseline_folders, folder_pattern, extra_predicate=pred
                )
        game_paths = sorted(game_paths)
        if not game_paths:
            print(f"Paths for game {game} not found.")
            continue
        print(f"Found {len(game_paths)} paths for game {game}.")
        methods_paths = organize_paths_by_method(
            game_paths,
            skip_methods=args.skip_methods,
            only_methods=args.only_methods,
        )
        if not methods_paths:
            print(f"No matching methods for game {game}; skipping.")
            continue
        game_method_paths[game] = methods_paths
        print(f"Game: {game}, Methods: {list(methods_paths.keys())}")

    for game, methods in game_method_paths.items():
        print(f"Game: {game}, Methods: {list(methods.keys())}")

    # Keep methods common to all games, or with --common_envs, games that have every method.
    ignore_random_policy = True
    if not game_method_paths:
        raise ValueError(
            "No games with matching methods found. Check --runs_dir and the "
            "--only_methods/--skip_methods patterns."
        )
    method_sets = [set(methods.keys()) for methods in game_method_paths.values()]
    if args.common_envs:
        common_methods = set.union(*method_sets)
    else:
        common_methods = set.intersection(*method_sets)
    if ignore_random_policy:
        common_methods.discard("random_policy")
        common_methods.discard("Random Policy")
        common_methods.discard("ppo-rnd (r_rnd)")
    if args.common_envs:
        dropped_games = [
            game
            for game, methods_paths in game_method_paths.items()
            if not common_methods <= set(methods_paths.keys())
        ]
        for game in dropped_games:
            del game_method_paths[game]
        if dropped_games:
            print(f"Dropped games missing some methods: {sorted(dropped_games)}")
        if not game_method_paths:
            raise ValueError(
                "No environment has all the selected methods: "
                f"{sorted(common_methods)}. Narrow the selection with "
                "--only_methods/--skip_methods."
            )
        games = sorted(game_method_paths)
    for game in game_method_paths:
        game_method_paths[game] = {
            method: paths
            for method, paths in game_method_paths[game].items()
            if method in common_methods
        }
    print("Common methods across all games:", sorted(common_methods))
    print("Game method paths after filtering:")
    for game, methods in game_method_paths.items():
        print(f"Game: {game}, Methods: {list(methods.keys())}")

    print("Initializing cache...")
    cache_dir = os.path.join(args.runs_dir, ".cache")
    cache = MultiRunDataCache(args.runs_dir)
    print("Cache initialized at:", cache_dir)

    # window=100 averages the last 100 episodes, window=0 the whole run.
    for window in (100, 0):
        print("=" * 90)
        print(f"Aggregating with window={window}")
        print("=" * 90)
        plot_for_window(
            args,
            game_method_paths,
            cache,
            baseline_folders,
            settings,
            games,
            domain,
            window,
        )
        plt.close("all")

    if args.plot_all:
        run_compare_methods(args)


def plot_for_window(
    args, game_method_paths, cache, baseline_folders, settings, games, domain, window
):
    """Normalize with one averaging window, run the tests and write the plots."""
    random_policy_scores = get_random_policy_scores(
        args.runs_dir,
        use_median=True,
        window=window,
        cache=cache,
        baseline_folders=baseline_folders,
    )
    print("Normalizing scores using random policy scores")
    normalized_scores, methods = normalize_method_scores(
        game_method_paths,
        random_policy_scores,
        map_names=False,
        cache=cache,
        window=window,
    )
    print("Normalized Scores: ", normalized_scores)

    cache.save()

    all_algorithms = sorted(methods)
    method_score_dict = {method: [] for method in all_algorithms}

    methods = set()
    for game in game_method_paths:
        for method in all_algorithms:
            scores = normalized_scores[game].get(method, [])
            if not len(settings) or method_name_to_setting(method) in settings:
                method_score_dict[method].append(scores)
                methods.add(method)
            else:
                method_score_dict.pop(method, None)
                print(
                    f"Skipping method {method} for game {game} due to settings filter."
                )
    all_algorithms = sorted(
        methods,
        reverse=True,
        key=lambda x: (
            x.replace("non-shared", "t").replace("DCPG ", "$e").replace("DCPG", "$d")
        ),
    )
    print("All Algorithms:", all_algorithms)

    for method in method_score_dict:
        min_seeds = min(
            len(scores) for scores in method_score_dict[method] if len(scores) > 0
        )
        print(f"Method: {method}: truncating to {min_seeds} seeds")
        method_score_dict[method] = [
            scores[:min_seeds] for scores in method_score_dict[method]
        ]
        # Shape (num_runs, num_games).
        method_score_dict[method] = np.array(method_score_dict[method]).T
        print(
            f"\tScores: {method_score_dict[method]}, shape: {method_score_dict[method].shape}"
        )

    methods = list(method_score_dict.keys())
    n_envs = method_score_dict[methods[0]].shape[1]
    n_seeds = method_score_dict[methods[0]].shape[0]

    print(
        f"\n🔍 Comparing {len(methods)} methods across {n_envs} environments and {n_seeds} seeds each."
    )
    print("=" * 90)

    significant_pairs = []
    alpha = 0.05
    print("Performing statistical comparisons using Mann-Whitney U test:")
    print("alpha =", alpha)

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            A, B = methods[i], methods[j]
            print(f"\n▶ Comparison: {A}  vs  {B}")
            print("-" * 90)

            p_values_AB = []
            p_values_BA = []
            better_counts = []

            for env_idx in range(n_envs):
                a_scores = method_score_dict[A][:, env_idx]
                b_scores = method_score_dict[B][:, env_idx]

                _, p_AB = mannwhitneyu(a_scores, b_scores, alternative="greater")
                _, p_BA = mannwhitneyu(b_scores, a_scores, alternative="greater")

                p_values_AB.append(p_AB)
                p_values_BA.append(p_BA)
                min_len = min(len(a_scores), len(b_scores))
                a_scores = a_scores[:min_len]
                b_scores = b_scores[:min_len]
                better_counts.append(np.mean(a_scores > b_scores))

                print(
                    f"  Env {env_idx + 1}: p(A>B)={p_AB:.4f}, p(B>A)={p_BA:.4f}, P(A>B)={np.mean(a_scores > b_scores):.2f}"
                )

            if n_envs == 1:
                p_combined_AB = p_values_AB[0]
                p_combined_BA = p_values_BA[0]
            else:
                chi2_AB, p_combined_AB = combine_pvalues(p_values_AB, method="fisher")
                chi2_BA, p_combined_BA = combine_pvalues(p_values_BA, method="fisher")

            mean_prob = np.mean(better_counts)

            print(
                f"\n  → Combined (Fisher) p(A>B)={p_combined_AB:.5f}, p(B>A)={p_combined_BA:.5f}"
            )
            print(f"  → Mean probability P(A>B)={mean_prob:.2f}")

            if p_combined_AB < alpha:
                print(
                    f"  ✅ Statistically significant: {A} > {B} (p={p_combined_AB:.4f})"
                )
                significant_pairs.append((A, B, p_combined_AB))
            elif p_combined_BA < alpha:
                print(
                    f"  ✅ Statistically significant: {B} > {A} (p={p_combined_BA:.4f})"
                )
                significant_pairs.append((B, A, p_combined_BA))
            else:
                print("  ✗ No significant difference")

    print("\n" + "=" * 90)
    print("✅ SUMMARY OF SIGNIFICANT RESULTS (p < 0.05)")
    print("=" * 90)

    if len(significant_pairs) == 0:
        print("No statistically significant differences found.")
    else:
        for winner, loser, pval in sorted(significant_pairs, key=lambda x: x[2]):
            print(f"  {winner:30s} > {loser:30s}  (p = {pval:.5f})")

    print("=" * 90)
    print("Statistical comparisons complete.\n")

    # Plot 1: aggregate metrics
    def aggregate_func(x):
        return np.array(
            [
                metrics.aggregate_median(x),
                metrics.aggregate_iqm(x),
                metrics.aggregate_mean(x),
                metrics.aggregate_optimality_gap(x),
            ]
        )

    indices_to_use = [2]
    metric_names = ["Median", "IQM", "Mean", "Optimality Gap"]
    metrics_chosen = [metric_names[i] for i in indices_to_use]

    dual = False
    if not dual:
        aggregate_scores, aggregate_cis = rly.get_interval_estimates(
            method_score_dict, aggregate_func, reps=50000
        )

        aggregate_scores = {
            method: scores[indices_to_use]
            for method, scores in aggregate_scores.items()
        }
        aggregate_cis = {
            method: cis[:, indices_to_use] for method, cis in aggregate_cis.items()
        }
    else:
        metrics_chosen = ["MuJoCo", "Atari"]

        aggregate_scores = {
            r"PPO ($r^\mathrm{ex}$)": np.array([0.6536732, 0.8535403]),
            "PPO-LIRPG ($r^\\mathrm{ex} + r^\\mathrm{in}$)": np.array(
                [0.67899345, 0.84807343]
            ),
            "PPO-LIRPG ($r^\\mathrm{in}$)": np.array([0.53887507, 0.1634703]),
        }

        aggregate_cis = {
            r"PPO ($r^\mathrm{ex}$)": np.array(
                [[0.61427308, 0.78999213], [0.69302815, 0.90858928]]
            ),
            "PPO-LIRPG ($r^\\mathrm{ex} + r^\\mathrm{in}$)": np.array(
                [[0.63004861, 0.79559062], [0.72568122, 0.90921431]]
            ),
            "PPO-LIRPG ($r^\\mathrm{in}$)": np.array(
                [[0.49063956, 0.08707913], [0.58860827, 0.23689673]]
            ),
        }

    print("Aggregate Scores:", aggregate_scores)
    print("Aggregate CIs:", aggregate_cis)
    print("Algorithms:", all_algorithms)
    color_palette = "tab20" if args.coupled or len(all_algorithms) > 10 else "tab10"
    colors = sns.color_palette(color_palette)
    if len(all_algorithms) > len(colors):
        colors += sns.color_palette("hls", len(all_algorithms) - len(colors))
    colors = colors[args.color_offset : args.color_offset + len(all_algorithms)][::-1]
    colors = dict(zip(all_algorithms, colors))
    print("Colors:", colors)
    if args.vertical:
        plot_interval_fn = plot_interval_estimates_vertical
    else:
        plot_interval_fn = plot_utils.plot_interval_estimates
    hatches = (
        ["//", ""] * int(len(all_algorithms) / 2)
        if args.coupled
        else [""] * len(all_algorithms)
    )
    fig, axs = plot_interval_fn(
        aggregate_scores,
        aggregate_cis,
        metric_names=metrics_chosen,
        algorithms=all_algorithms,
        subfigure_width=4,
        row_height=1,
        xlabel="Normalized Score" if args.vertical else "",
        colors=colors,
        hatches=hatches,
        legend_location="right",
    )
    if not args.vertical:
        plt.xlabel("Normalized Score", labelpad=10, fontsize=18)

    if len(metrics_chosen) == 1 and not args.vertical:
        axs = [axs]
    for ax in axs:
        ax_min = min(
            min(ci[0] for ci in aggregate_cis[method]) for method in aggregate_cis
        )
        ax_min = min(ax_min, 0)
        if args.vertical:
            ax.set_ylim(ax_min, 1.1)
        else:
            ax.set_xlim(ax_min, 1)
    fig.set_size_inches(len(metrics_chosen) * (len(all_algorithms) // 2 + 0.5), 4)

    for i in range(1, len(axs)):
        if not args.vertical:
            axs[i].axvline(x=0, color="black", linestyle="--", linewidth=0.5)

    if args.coupled:
        for ax in axs:
            for i in range(1, len(all_algorithms), 2):
                if not args.vertical:
                    ax.axhline(y=i - 1.5, color="gray", linestyle="--", linewidth=0.5)
                else:
                    ax.axvline(
                        x=i * 0.18 - 0.72, color="gray", linestyle="--", linewidth=0.5
                    )

    fig_path = os.path.join(args.output_dir, f"aggregate_metrics-w={window}.png")
    fig.savefig(fig_path, bbox_inches="tight")
    pdf_path = fig_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved aggregate metrics plot to {fig_path}")

    # Plot 2: probability of improvement
    if args.n_plots < 2:
        print("Skipping remaining plots as n_plots < 2.")
        return
    algorithm_pairs = {}
    for i, a1 in enumerate(all_algorithms):
        for a2 in all_algorithms[i + 1 :]:
            algorithm_pairs[f"{a1},{a2}"] = (
                method_score_dict[a1],
                method_score_dict[a2],
            )

    avg_probs, avg_prob_cis = rly.get_interval_estimates(
        algorithm_pairs, metrics.probability_of_improvement, reps=2000
    )

    ax = plot_utils.plot_probability_of_improvement(
        avg_probs,
        avg_prob_cis,
        figsize=(8, len(algorithm_pairs) / 2 + 1),
    )
    ax.set_xlim(0, 1)

    fig_path = os.path.join(
        args.output_dir, f"probability_of_improvement-w={window}.png"
    )
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    pdf_path = fig_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved probability of improvement plot to {fig_path}")

    # Plot 3: sample efficiency curve, scores shaped (num_runs, num_games, num_frames)
    if args.n_plots < 3:
        print("Skipping remaining plots as n_plots < 3.")
        return
    ale_frames_scores_dict = get_all_frames_scores_dict(game_method_paths)
    print("ALE Frames Scores Dict:")
    for key, value in ale_frames_scores_dict.items():
        print(f"{key}: {value.shape}")
        num_frames = value.shape[-1]

    frames = np.arange(num_frames)

    def iqm(scores):
        return np.array(
            [
                metrics.aggregate_iqm(scores[..., frame])
                for frame in range(scores.shape[-1])
            ]
        )

    iqm_scores, iqm_cis = rly.get_interval_estimates(
        ale_frames_scores_dict, iqm, reps=50000
    )

    ax = plot_utils.plot_sample_efficiency_curve(
        frames + 1,
        iqm_scores,
        iqm_cis,
        algorithms=all_algorithms,
        xlabel="Number of Frames (in millions)",
        ylabel="IQM Random Normalized Score",
    )

    ax.set_title(f"Sample Efficiency Curve for {domain} on {len(games)} Games")

    plt.legend()
    plt.savefig(
        os.path.join(args.output_dir, f"sample_efficiency_curve-w={window}.png"),
        bbox_inches="tight",
    )

    # Plot 4: performance profiles
    if args.n_plots < 4:
        print("Skipping remaining plots as n_plots < 4.")
        return
    thresholds = np.linspace(0.0, 8.0, 81)
    perf_profiles, perf_cis = rly.create_performance_profile(
        method_score_dict, thresholds
    )

    fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    plot_utils.plot_performance_profiles(
        perf_profiles,
        thresholds,
        performance_profile_cis=perf_cis,
        colors=dict(zip(all_algorithms, sns.color_palette("colorblind"))),
        xlabel=r"Random Normalized Score $(\tau)$",
        ax=ax,
    )
    fig.tight_layout()
    plt.legend()
    fig.savefig(
        os.path.join(args.output_dir, f"performance_profiles-w={window}.png"),
        bbox_inches="tight",
    )

    fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    plot_utils.plot_performance_profiles(
        perf_profiles,
        thresholds,
        performance_profile_cis=perf_cis,
        use_non_linear_scaling=True,
        xticks=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
        colors=dict(zip(all_algorithms, sns.color_palette("colorblind"))),
        xlabel=r"Random Normalized Score $(\tau)$",
        ax=ax,
    )
    fig.tight_layout()
    plt.legend()
    fig.savefig(
        os.path.join(args.output_dir, f"performance_profiles_nonlinear-w={window}.png"),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
