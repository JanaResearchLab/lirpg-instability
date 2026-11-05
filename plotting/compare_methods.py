import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from common_utils import (
    MultiRunDataCache,
    RunDataCache,
    baseline_event_paths,
    detect_games,
    get_all_postfixes,
    get_random_policy_scores,
    load_baseline_paths,
    method_name_mapping,
    method_name_to_setting,
    normalize_method_scores,
    organize_paths_by_method,
    shorten_env_name,
)
from scipy.interpolate import interp1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update({"font.size": 23})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare different methods for RL experiments."
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="runs",
        help="Directory containing the runs.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Window size for the moving average.",
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
        default=None,
        help="Comma-separated settings to include.",
    )
    parser.add_argument(
        "--skip_methods",
        type=str,
        default=None,
        help="Comma-separated methods to skip (fnmatch wildcards; aliases mix, in, ex).",
    )
    parser.add_argument(
        "--only_methods",
        type=str,
        default=None,
        help="Comma-separated methods to keep (fnmatch wildcards; aliases mix, in, ex).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory (default: <runs_dir>/.cache).",
    )
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="Force reload all data, ignoring cache.",
    )
    parser.add_argument(
        "--coupled",
        action="store_true",
        help="Pair adjacent methods by color and hatch the second of each pair.",
    )
    parser.add_argument(
        "--legend_location",
        type=str,
        default="right",
        choices=["below", "right", "none"],
        help="Legend position, or none to hide it.",
    )
    return parser.parse_args()


def moving_average(data, window_size):
    window_size = min(window_size, len(data))
    if window_size < 1:
        return data
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode="valid")


def load_tf_events_data(
    event_paths,
    metric: str = "charts/episodic_return",
    cache: RunDataCache = None,
    force_reload=False,
):
    """Return (steps, values, episodes) per event file, using the cache when possible."""
    data = []
    print(
        f"Loading data for metric '{metric}' from {len(event_paths)} paths... {cache=} {force_reload=}"
    )

    for path in event_paths:
        if not force_reload and cache is not None:
            cached_data = cache.get_cached_data(path, metric)
            if cached_data is not None:
                print(f"Loading cached data for {path}")
                steps, values, episodes = cached_data
                data.append((steps, values, episodes))
                continue

        print(f"Loading fresh data for {path}")
        event_acc = EventAccumulator(path, size_guidance={"scalars": 0})
        event_acc.Reload()

        try:
            scalar_data = event_acc.Scalars(metric)
            steps = [x.step for x in scalar_data]
            values = [x.value for x in scalar_data]
            episodes = list(range(1, len(values) + 1))

            if cache is not None:
                cache.cache_data(path, steps, values, episodes, metric=metric)

            data.append((steps, values, episodes))

        except KeyError:
            print(f"Key '{metric}' not found in {path}")

    return data


def interpolate_data(seed_data, common_steps):
    """Interpolate metric values for a seed at common timesteps."""
    steps, values, _ = seed_data
    steps = np.array(steps)
    values = np.array(values)
    unique_steps, inverse = np.unique(steps, return_inverse=True)
    if len(unique_steps) < len(steps):
        avg_values = np.zeros(len(unique_steps))
        for i in range(len(unique_steps)):
            avg_values[i] = np.mean(values[inverse == i])
        steps, values = unique_steps, avg_values
    interp_func = interp1d(steps, values, kind="linear", fill_value="extrapolate")
    return interp_func(common_steps)


def exponential_moving_average(data, smoothing_factor=0.9):
    """TensorBoard-style exponential smoothing."""
    alpha = 1 - smoothing_factor
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]

    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]

    return smoothed


def process_method_data(data, window_size):
    """Interpolate seeds onto common steps, smooth, and return (steps, mean, std, n_seeds)."""
    if not data:
        print("No data to process")
        return None, None, None, None

    all_steps = [d[0] for d in data]

    if not all_steps or any(len(steps) == 0 for steps in all_steps):
        print("Warning: Some seeds have no step data")
        return None, None, None, None

    common_steps = np.linspace(
        max(min(steps) for steps in all_steps if len(steps) > 0),
        min(max(steps) for steps in all_steps if len(steps) > 0),
        num=1000,
    )

    if len(common_steps) == 0:
        print("Warning: No common steps found across seeds")
        return np.array([]), np.array([]), np.array([]), len(data)

    interpolated_values = []
    for d in data:
        interp_values = interpolate_data(d, common_steps)
        interpolated_values.append(interp_values)

    if not interpolated_values:
        print("Warning: No interpolated values")
        return np.array([]), np.array([]), np.array([]), len(data)

    interpolated_values = np.array(interpolated_values)

    smoothed_values = []
    for seed_values in interpolated_values:
        if len(seed_values) >= window_size:
            smoothed = exponential_moving_average(seed_values)
            smoothed_values.append(smoothed)
        else:
            print(
                f"Warning: Seed data length {len(seed_values)} < window size {window_size}"
            )
            if len(seed_values) > 0:
                smoothed = exponential_moving_average(seed_values)
                smoothed_values.append(smoothed)

    if not smoothed_values:
        print("Warning: No smoothed values")
        return np.array([]), np.array([]), np.array([]), len(data)

    smoothed_values = np.array(smoothed_values)
    smoothed_steps = common_steps

    mean_values = np.mean(smoothed_values, axis=0)
    std_values = np.std(smoothed_values, axis=0)

    if len(smoothed_steps) != len(mean_values):
        print(
            f"Warning: Dimension mismatch - steps: {len(smoothed_steps)}, means: {len(mean_values)}"
        )
        min_len = min(len(smoothed_steps), len(mean_values))
        smoothed_steps = smoothed_steps[:min_len]
        mean_values = mean_values[:min_len]
        std_values = std_values[:min_len]

    return smoothed_steps, mean_values, std_values, len(data)


def plot_methods_comparison(ax, methods_data, game, xlabel, ylabel):
    colors = plt.cm.tab10.colors

    methods_data = dict(sorted(methods_data.items()))

    for i, (method, (steps, mean, std, num_seeds)) in enumerate(methods_data.items()):
        method = method_name_mapping.get(method, method)
        if (
            method == "Random Policy"
            or method == "random_policy"
            or method == "PPO-RND (R_rnd only)"
        ):
            continue
        if len(steps) == 0 or len(mean) == 0:
            print(f"Skipping method {method} due to empty data")
            continue

        if len(steps) != len(mean):
            print(
                f"Warning: Dimension mismatch for {method} - steps: {len(steps)}, means: {len(mean)}"
            )
            min_len = min(len(steps), len(mean))
            steps = steps[:min_len]
            mean = mean[:min_len]
            std = std[:min_len] if len(std) >= min_len else std[: len(std)]

        color = colors[i % len(colors)]
        label = f"{method} (n={num_seeds})"
        ax.plot(steps, mean, label=label, color=color)

        if len(std) == len(steps):
            ax.fill_between(steps, mean - std, mean + std, alpha=0.3, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{shorten_env_name(game)} - {ylabel} vs {xlabel}")
    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    ax.grid(True)


def process_episode_data(data, window_size):
    if not data or len(data) == 0:
        return np.array([]), np.array([]), np.array([])

    min_episodes = min(len(d[2]) for d in data)
    if min_episodes == 0:
        return np.array([]), np.array([]), np.array([])

    aligned_episodes = data[0][2][:min_episodes]
    aligned_values = np.array([d[1][:min_episodes] for d in data])

    smoothed_episode_values = []
    for seed_values in aligned_values:
        if len(seed_values) >= window_size:
            smoothed = moving_average(seed_values, window_size)
            smoothed_episode_values.append(smoothed)
        elif len(seed_values) > 0:
            smaller_window = max(1, len(seed_values) // 2)
            smoothed = moving_average(seed_values, smaller_window)
            smoothed_episode_values.append(smoothed)

    if not smoothed_episode_values:
        return np.array([]), np.array([]), np.array([])

    min_length = min(len(values) for values in smoothed_episode_values)
    smoothed_episode_values = np.array(
        [values[:min_length] for values in smoothed_episode_values]
    )

    smoothed_episodes = aligned_episodes[window_size - 1 : window_size - 1 + min_length]

    mean_episode_values = np.mean(smoothed_episode_values, axis=0)
    std_episode_values = np.std(smoothed_episode_values, axis=0)

    if len(smoothed_episodes) != len(mean_episode_values):
        min_len = min(len(smoothed_episodes), len(mean_episode_values))
        smoothed_episodes = smoothed_episodes[:min_len]
        mean_episode_values = mean_episode_values[:min_len]
        std_episode_values = std_episode_values[:min_len]

    return smoothed_episodes, mean_episode_values, std_episode_values


def plot_interval_plot(
    method_paths,
    random_policy_scores=None,
    use_median=True,
    cache: RunDataCache = None,
    window: int = 100,
    coupled: bool = False,
    legend_location: str = "below",
):
    """Bar plot of normalized score spread per game and method."""

    normalized_scores, methods = normalize_method_scores(
        method_paths, random_policy_scores, cache=cache, window=window
    )

    fig, ax = plt.subplots(figsize=(12, 4))

    games = list(method_paths.keys())
    methods = sorted(methods)
    if coupled:
        method_colors = {m: plt.cm.tab20(i % 20) for i, m in enumerate(methods)}
        method_hatches = {
            m: "////" if i % 2 == 1 else "" for i, m in enumerate(methods)
        }
    else:
        method_colors = {m: plt.cm.tab10(i % 10) for i, m in enumerate(methods)}
        method_hatches = {m: "" for m in methods}

    game_positions = np.arange(len(games))
    width = 0.8 / len(methods)

    for j, method in enumerate(methods):
        if (
            method == "Random Policy"
            or method == "random_policy"
            or method == "PPO-RND (R_rnd only)"
        ):
            continue
        offsets = game_positions - 0.4 + j * width + width / 2

        for i, game in enumerate(games):
            if method not in normalized_scores[game]:
                continue
            norm_scores = normalized_scores[game][method]
            if norm_scores:
                norm_scores = np.array(norm_scores)
                if use_median:
                    center = np.median(norm_scores)
                    low = np.percentile(norm_scores, 25)
                    high = np.percentile(norm_scores, 75)
                else:
                    center = np.mean(norm_scores)
                    print(f"Method: {method}, Game: {game}, Reward: {center}")
                    std = np.std(norm_scores)
                    low = center - std
                    high = center + std

                ax.bar(
                    offsets[i],
                    high - low,
                    bottom=low,
                    width=width - 0.2 / len(methods),
                    color=method_colors[method],
                    alpha=0.7,
                    hatch=method_hatches[method],
                    label=method if i == 0 else "",
                )
                ax.hlines(
                    center,
                    offsets[i] - width / 4,
                    offsets[i] + width / 4,
                    color="black",
                    linewidth=2,
                )

    for pos in game_positions[:-1]:
        ax.axvline(x=pos + 0.5, color="gray", linestyle="--", linewidth=0.5)

    ax.set_xticks(game_positions)
    game_names = [shorten_env_name(game) for game in games]
    ax.set_xticklabels(game_names, rotation=30, ha="right")
    ax.set_ylabel("Normalized Score")
    ax.set_yticks(np.arange(0, 1.05, 0.2))
    ax.tick_params(axis="y", labelsize=18)
    if legend_location == "right":
        ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")
    elif legend_location == "below":
        ax.legend(bbox_to_anchor=(0.5, -0.4), loc="upper center", ncol=3)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()


def main():
    args = parse_args()

    if args.cache_dir is None:
        cache_dir = os.path.join(args.runs_dir, ".cache")
    else:
        cache_dir = args.cache_dir

    cache = MultiRunDataCache(args.runs_dir, primary_cache_dir=args.cache_dir)
    print(f"Using cache directory: {cache_dir}")

    baseline_folders = load_baseline_paths(args.runs_dir)

    games = detect_games(args.runs_dir)

    figures_dir = os.path.join(args.runs_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    if args.postfixes == "all":
        postfixes = sorted(get_all_postfixes(args.runs_dir))
    else:
        postfixes = args.postfixes.split(",") if args.postfixes else None
    settings = args.settings.split(",") if args.settings else None

    skip_methods = args.skip_methods.split(",") if args.skip_methods else None
    only_methods = args.only_methods.split(",") if args.only_methods else None

    game_method_paths = {}
    for game in games:
        game_paths = []
        if args.postfixes is None:
            folder_pattern = f"{game}*--*--*"
            game_paths = sorted(
                glob(
                    os.path.join(args.runs_dir, folder_pattern, "events.out.tfevents.*")
                )
            )
            game_paths += baseline_event_paths(baseline_folders, folder_pattern)
        else:
            for postfix in postfixes:
                folder_pattern = f"{game}*--*_*_0{postfix}--*"  # r_rnd = 0 runs only
                game_paths += sorted(
                    glob(
                        os.path.join(
                            args.runs_dir,
                            folder_pattern,
                            "events.out.tfevents.*",
                        )
                    )
                )
                game_paths += baseline_event_paths(baseline_folders, folder_pattern)

        # Postfix globs can overlap (ppg matches ppg_ent_0.01_vreg_0).
        game_paths = sorted(set(game_paths))

        if len(game_paths) == 0:
            continue

        methods_paths = organize_paths_by_method(
            game_paths, skip_methods=skip_methods, only_methods=only_methods
        )
        if settings:
            methods_paths = {
                method: paths
                for method, paths in methods_paths.items()
                if method_name_to_setting(method) in settings
            }
        game_method_paths[game] = methods_paths

        if not methods_paths:
            print(f"No methods found for game {game}.")
            continue

        print(
            f"Found {len(methods_paths)} methods for {game}: {', '.join(methods_paths.keys())}"
        )

        methods_data_steps = {}

        for method, paths in methods_paths.items():
            print(f"Processing {method} with {len(paths)} seeds")
            data = load_tf_events_data(
                paths, cache=cache, force_reload=args.force_reload
            )
            if data:
                steps, mean, std, num_seeds = process_method_data(
                    data, args.window_size
                )
                if steps is not None:
                    min_len = min(len(steps), len(mean), len(std))
                    methods_data_steps[method] = (
                        steps[:min_len],
                        mean[:min_len],
                        std[:min_len],
                        num_seeds,
                    )
                    print(f"  Processed {method}: {len(steps[:min_len])} data points")

        if not methods_data_steps:
            print(f"No valid data found for game {game}.")
            continue

        valid_methods = {m: d for m, d in methods_data_steps.items() if len(d[0]) > 0}
        if not valid_methods:
            print(f"No methods with valid data for game {game}.")
            continue

        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        plot_methods_comparison(
            ax, valid_methods, game, xlabel="Global Step", ylabel="Episodic Return"
        )
        plot_path = os.path.join(figures_dir, f"{game}_methods_comparison.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved comparison plot for {game} at {plot_path}")

        methods_data_episodes = {}

        for method, paths in methods_paths.items():
            data = load_tf_events_data(
                paths, cache=cache, force_reload=args.force_reload
            )
            if not data:
                continue

            smoothed_episodes, mean_episode_values, std_episode_values = (
                process_episode_data(data, args.window_size)
            )

            if len(smoothed_episodes) > 0 and len(mean_episode_values) > 0:
                min_len = min(
                    len(smoothed_episodes),
                    len(mean_episode_values),
                    len(std_episode_values),
                )
                methods_data_episodes[method] = (
                    smoothed_episodes[:min_len],
                    mean_episode_values[:min_len],
                    std_episode_values[:min_len],
                    len(data),
                )
                print(
                    f"  Processed {method} episodes: {len(smoothed_episodes[:min_len])} data points"
                )

        valid_episode_methods = {
            m: d for m, d in methods_data_episodes.items() if len(d[0]) > 0
        }

        if valid_episode_methods:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
            plot_methods_comparison(
                ax,
                valid_episode_methods,
                game,
                xlabel="Episode",
                ylabel="Episodic Return",
            )
            plot_path = os.path.join(
                figures_dir, f"{game}_methods_episodes_comparison.png"
            )
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"Saved episode comparison plot for {game} at {plot_path}")

    cache.save()
    print(f"Cache saved to {cache.cache_file}")

    random_policy_scores = get_random_policy_scores(
        args.runs_dir, use_median=True, cache=cache, baseline_folders=baseline_folders
    )
    if not random_policy_scores:
        print("No random policy scores found. Plotting without normalizing...")

    for use_median in [True, False]:
        for window in [100, 0]:
            print(
                f"Plotting interval plot with use_median={use_median}, window={window}"
            )
            plot_interval_plot(
                game_method_paths,
                random_policy_scores,
                cache=cache,
                window=window,
                use_median=use_median,
                coupled=args.coupled,
                legend_location=args.legend_location,
            )
            use_median_str = "median" if use_median else "mean"
            plot_path = os.path.join(
                figures_dir,
                f"interval_plot-{use_median_str}-w={window}.png",
            )
            plt.savefig(plot_path, bbox_inches="tight")
            plot_path_pdf = plot_path.replace(".png", ".pdf")
            plt.savefig(plot_path_pdf, bbox_inches="tight")
            plt.close()
            print(f"Saved interval plot at {plot_path}")


if __name__ == "__main__":
    main()
