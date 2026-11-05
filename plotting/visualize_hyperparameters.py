"""Figures 3 and 8: normalized score against lambda_in per game; also writes best_runs.json.

Usage: python plotting/visualize_hyperparameters.py --runs_dir runs-atari-tuning --postfixes=""
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from common_utils import (
    RunDataCache,
    get_all_postfixes,
    get_avg_return_from_event_file,
    get_random_policy_scores,
    load_baseline_paths,
    load_score_bounds,
    shorten_env_name,
)
from tqdm import tqdm

plt.rcParams.update({"font.size": 20})


# lambda_in candidates for the mixed-reward selection of Tables 4-5 (Appendix B.3)
SELECTION_LAMBDAS = (0.01, 0.1, 0.3, 0.7, 0.9, 0.99)


def parse_run_folder_name(folder_name: str):
    pattern = r"(?P<env>[^/]+)__--tuning_(?P<r_ex>[\d.]+)_(?P<r_in>[\d.]+)_(?P<r_rnd>[\d.]+)--__(?P<seed>\d+)__"
    match = re.search(pattern, folder_name)
    if not match:
        return None
    return {
        "env": match.group("env"),
        "r_ex": float(match.group("r_ex")),
        "r_in": float(match.group("r_in")),
        "r_rnd": float(match.group("r_rnd")),
        "seed": int(match.group("seed")),
    }


def normalize_scores(scores_by_env, random_policy_scores, bounds=None):
    """{(env, method): score scaled so the floor is 0 and the ceiling is 1}."""
    if bounds is None:
        bounds = load_score_bounds()
    normalized = {}
    for env, scores in scores_by_env.items():
        stored = bounds.get(env, {})
        rand_score = stored.get("min", random_policy_scores.get(env))
        vals = list(scores.values())
        if rand_score is None:
            rand_score = min(vals) if len(vals) else None
        max_score = max(vals) if len(vals) else None
        if max_score is not None and "max" in stored:
            max_score = max(max_score, stored["max"])
        if max_score is None or max_score == rand_score:
            continue
        for k, v in scores.items():
            if v is not None:
                normalized[(env, k)] = (v - rand_score) / (max_score - rand_score)
    return normalized


def visualize_lineplot(data, x_key, title, out_path: str = None):
    """Plot per-env and overall normalized score against x_key."""
    plt.figure(figsize=(10, 6))

    for line_name, marker, linestyle, alpha, color, x_vals, scores, linewidth in data:
        label = line_name if line_name == "Overall" else shorten_env_name(line_name)
        plt.plot(
            x_vals,
            scores,
            label=label,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            color=color,
        )

    plt.xlabel(x_key)
    plt.ylabel("Normalized Score")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=22)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.ylim(0, 1)

    plt.tight_layout()
    if out_path is not None:
        print(f"Plot saved to {out_path}")
        plt.savefig(out_path)
        pdf_path = out_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()


def visualize_ternary_plot(data, title, aggregate_fn, out_path=None):
    for postfix, data_by_postfix in data.items():
        records = []
        for (rex, rin, rrnd), score in data_by_postfix.items():
            if score is not None:
                records.append(
                    {
                        "r_ex": rex,
                        "r_in": rin,
                        "r_rnd": rrnd,
                        "score": aggregate_fn(list(score.values())),
                    }
                )
        df = pd.DataFrame(records)

        fig = px.scatter_ternary(
            df,
            a="r_ex",
            b="r_in",
            c="r_rnd",
            color="score",
            color_continuous_scale="Viridis",
            size_max=10,
            title=title,
        )

        fig.update_layout(coloraxis_colorbar={"title": "Score"})

        if out_path:
            out_path = out_path.replace(".html", postfix + ".html")
            fig.write_html(out_path)
        else:
            fig.show()


def save_csv(data, keys, out_path):
    with open(out_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([*keys, "normalized_score"])
        for k, v in data.items():
            if v is not None:
                writer.writerow([*k, v])


def process_2d_setting(method_name, data, aggregate_fn, output_dir, x_idx, y_idx):
    """Plot a two-coefficient sweep; data: {postfix: {(r_ex, r_in, r_rnd): {env: score}}}."""

    keys = [
        r"$r^\mathrm{ex}$",
        "$r^\\mathrm{in}$ coefficient ($\\lambda$)",
        r"$r^\mathrm{rnd}$",
    ]
    csv_keys = [keys[x_idx], keys[y_idx]]
    x_key = keys[x_idx]

    all_postfixes = sorted(data.keys())
    all_envs = set()
    env_scores_by_params = {postfix: defaultdict(dict) for postfix in all_postfixes}
    agg_scores_by_env = {postfix: defaultdict(list) for postfix in all_postfixes}
    for postfix, data_by_postfix in data.items():
        csv_path = os.path.join(output_dir, f"{method_name}{postfix}.csv")
        processed_data = {}
        if len(set([params[x_idx] for params in data_by_postfix])) == 1:
            x_idx, y_idx = y_idx, x_idx
            x_key = keys[x_idx]
        for params, scores_by_env in data_by_postfix.items():
            new_key = (params[x_idx], params[y_idx])
            all_param_scores = []
            for env, score in scores_by_env.items():
                all_param_scores.append(score)
                if env not in env_scores_by_params[postfix]:
                    env_scores_by_params[postfix][env] = {"x_vals": [], "scores": []}
                env_scores_by_params[postfix][env]["x_vals"].append(params[x_idx])
                env_scores_by_params[postfix][env]["scores"].append(score)
                all_envs.add(env)
            print(f"Processing {postfix} {params} with scores: {all_param_scores}")
            processed_data[new_key] = aggregate_fn(all_param_scores)
            agg_scores_by_env[postfix]["x_vals"].append(params[x_idx])
            agg_scores_by_env[postfix]["scores"].append(processed_data[new_key])

        save_csv(processed_data, csv_keys, csv_path)

    all_envs = sorted(all_envs)
    colors_postfix = [
        "navy",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "teal",
    ]
    postfix_colors = {
        postfix: color for postfix, color in zip(all_postfixes, colors_postfix)
    }
    linestyles = ["--", "-.", ":", " "]
    postfix_linestyles = {
        postfix: linestyle for postfix, linestyle in zip(all_postfixes, linestyles)
    }
    colors_envs = [
        "#1f77b4",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#ff7f0e",
        "#8c564b",
        "#e377c2",
        "#17becf",
    ]
    env_colors = {env: color for env, color in zip(all_envs, colors_envs)}
    line_plot_data = []
    for postfix in all_postfixes:
        linestyle = postfix_linestyles[postfix]
        for env, plot_data in sorted(env_scores_by_params[postfix].items()):
            line_name = env
            marker = "x"
            alpha = 0.6
            color = env_colors[env]
            x_vals = plot_data["x_vals"]
            scores = plot_data["scores"]

            sorted_indices = np.argsort(x_vals)
            x_vals = np.array(x_vals)[sorted_indices]
            scores = np.array(scores)[sorted_indices]

            line_plot_data.append(
                (line_name, marker, linestyle, alpha, color, x_vals, scores, 2)
            )
        line_name = "Overall"
        marker = "o"
        alpha = 1.0
        color = postfix_colors[postfix]
        x_vals = agg_scores_by_env[postfix]["x_vals"]
        scores = agg_scores_by_env[postfix]["scores"]
        sorted_indices = np.argsort(x_vals)
        x_vals = np.array(x_vals)[sorted_indices]
        scores = np.array(scores)[sorted_indices]
        line_plot_data.append((line_name, marker, "-", alpha, color, x_vals, scores, 3))

    visualize_lineplot(
        line_plot_data,
        x_key,
        f"{method_name.upper()}",
        os.path.join(output_dir, f"{method_name}.png"),
    )


def process_setting(name, data, aggregate_fn, output_dir):
    """Write the plots and CSV for one setting (PPO-RND, PPO-LIRPG or PPO-LIRPG-RND)."""
    if name == "PPO-RND":
        # x = r_rnd, y = r_ex
        process_2d_setting(name, data, aggregate_fn, output_dir, 2, 0)
    elif name == "PPO-LIRPG":
        # x = r_in, y = r_ex
        process_2d_setting(name, data, aggregate_fn, output_dir, 1, 0)
    elif name == "PPO-LIRPG-RND":
        csv_path = os.path.join(output_dir, f"{name}.csv")
        save_csv(data, ["r_ex", "r_in", "r_rnd"], csv_path)
        visualize_ternary_plot(
            data,
            f"{name.upper()}",
            aggregate_fn,
            os.path.join(output_dir, f"{name}.html"),
        )


def get_scores_by_config(configs, paths, cache: RunDataCache = None, window: int = 100):
    runs_by_config = defaultdict(list)

    for i, full_path in enumerate(tqdm(paths)):
        if not os.path.isdir(full_path):
            continue
        config = configs[i]
        key = config["key"]
        event_path = glob(os.path.join(full_path, "events.out.tfevents.*"))[0]
        avg_return = get_avg_return_from_event_file(
            event_path, cache=cache, window=window
        )
        if avg_return is not None:
            runs_by_config[(config["env"], key)].append(avg_return)
        else:
            print(
                f"[WARNING] skipping {full_path}, fewer than {window} episodes or no event data"
            )
    if cache:
        cache.save()
    return runs_by_config


def get_scores_by_env(paths, configs, use_median, cache: RunDataCache = None):
    runs_by_config = get_scores_by_config(configs, paths, cache)

    scores_by_env = defaultdict(dict)
    aggregation_fn = np.median if use_median else np.mean

    for (env, config), returns in runs_by_config.items():
        scores_by_env[env][config] = aggregation_fn(returns)
    return scores_by_env


def main(
    runs_dir,
    output_dir,
    use_median=False,
    postfixes: list[str] = [""],
    use_cache: bool = True,
):
    settings = {
        "PPO-RND (r_rnd)": lambda rex, rin, rrnd: rex == 0 and rin == 0 and rrnd > 0,
        "PPO-RND": lambda _, rin, rrnd: rin == 0 and rrnd > 0,
        "PPO-LIRPG (r_in)": lambda rex, rin, rrnd: rex == 0 and rrnd == 0 and rin > 0,
        "PPO-LIRPG": lambda rex, rin, rrnd: rex > 0 and rrnd == 0 and rin > 0,
        "PPO-LIRPG-RND": lambda rex, rin, rrnd: rex > 0 and rin > 0 and rrnd > 0,
        "PPO": lambda rex, rin, rrnd: rex > 0 and rin == 0 and rrnd == 0,
    }
    aggregation_fn = np.median if use_median else np.mean
    normalized_scores_by_setting = {
        name: defaultdict(lambda: defaultdict(dict)) for name in settings
    }
    for run_postfix in postfixes:
        print(f"Processing runs with postfix: {run_postfix}")
        if run_postfix and not output_dir.endswith(run_postfix):
            out_dir = output_dir + run_postfix
        else:
            out_dir = output_dir
        os.makedirs(out_dir, exist_ok=True)

        cache_dir = os.path.join(runs_dir, ".cache")
        cache = RunDataCache(cache_dir) if use_cache else None

        if run_postfix:
            paths = glob(os.path.join(runs_dir, f"*--tuning_*_*_*{run_postfix}--*"))
        else:
            paths_w_postfix = glob(os.path.join(runs_dir, "*--tuning_*_*_*--*"))
            to_skip = glob(os.path.join(runs_dir, "*--tuning_*_*_*-*--*"))
            paths = [p for p in paths_w_postfix if p not in to_skip]

        if len(paths) == 0:
            print(f"No paths found for postfix '{run_postfix}', skipping...")
            continue

        all_paths = {}
        configs = []
        for path in paths:
            if "Ant" in path or "Humanoid" in path:
                continue
            if not os.path.isdir(path):
                continue
            folder = os.path.basename(path)
            if run_postfix:
                folder = folder.replace(run_postfix, "")
            config = parse_run_folder_name(folder)
            if not config or config["r_rnd"] > 0:
                continue
            config["key"] = (config["r_ex"], config["r_in"], config["r_rnd"])
            all_paths[path] = os.path.getmtime(path)
            configs.append(config)

        scores_by_env = get_scores_by_env(all_paths, configs, use_median, cache=cache)
        print(f"Scores by env: {scores_by_env}")

        random_policy_scores = get_random_policy_scores(
            runs_dir,
            use_median,
            cache=cache,
            baseline_folders=load_baseline_paths(runs_dir),
        )
        print("Calculated random policy scores")

        normalized_scores = normalize_scores(scores_by_env, random_policy_scores)
        print(f"Normalized scores: {normalized_scores}")

        for (env, (rex, rin, rrnd)), score in normalized_scores.items():
            print(f"Processing {env} {rex} {rin} {rrnd} with score: {score}")
            for name, cond in settings.items():
                if cond(rex, rin, rrnd):
                    normalized_scores_by_setting[name][run_postfix][(rex, rin, rrnd)][
                        env
                    ] = score
        for name, postfix_data in normalized_scores_by_setting.items():
            if not postfix_data:
                print(f"No data for {name}")
                continue
            if not postfix_data[run_postfix]:
                print(f"No data for {name} with postfix {run_postfix}, skipping...")
                continue
            process_setting(
                name,
                {run_postfix: postfix_data[run_postfix]},
                aggregation_fn,
                out_dir,
            )
            print(f"Processed {name}")

    if len(postfixes) > 1:
        output_dir = output_dir + "-" + ",".join(postfixes)
    os.makedirs(output_dir, exist_ok=True)

    best_runs_path = os.path.join(runs_dir, "best_runs.json")
    if os.path.exists(best_runs_path):
        print("Found existing runs json")
        with open(best_runs_path, "r") as f:
            best_runs = json.load(f)
    else:
        best_runs = {}
    best_configs = {name: {} for name in settings}

    for name, postfix_data in normalized_scores_by_setting.items():
        if not postfix_data:
            print(f"No data for {name}")
            continue
        print(f"Processing settings for {name}")
        print(f"Postfix data: {postfix_data}")
        process_setting(
            name,
            postfix_data,
            aggregation_fn,
            output_dir,
        )
        print(f"Processed {name}")

        for postfix, data_by_postfix in postfix_data.items():
            best_configs[name][postfix] = {}
            for (rex, rin, rrnd), scores_by_env in data_by_postfix.items():
                if name == "PPO-LIRPG" and rin not in SELECTION_LAMBDAS:
                    continue
                for env, score in scores_by_env.items():
                    print(
                        f"Checking config {name} {postfix} {env} with config {(rex, rin, rrnd)} score {score}"
                    )
                    if env not in best_configs[name][postfix]:
                        best_configs[name][postfix][env] = (float("-inf"), tuple())
                    if score > best_configs[name][postfix][env][0]:
                        best_configs[name][postfix][env] = (score, (rex, rin, rrnd))

        for postfix, env_data in best_configs[name].items():
            for env, (score, config) in env_data.items():
                config = list(config)
                for i in range(3):
                    if config[i] == 1 or config[i] == 0:
                        config[i] = int(config[i])
                    elif i > 0:
                        config[i] = str(config[i]).replace("0.", ".").replace("0.0", "")
                wild_card_path = os.path.join(
                    runs_dir,
                    f"{env}__--tuning_{config[0]}_*{config[1]}_*{config[2]}{postfix}--*",
                )
                print(f"Searching for path: {wild_card_path}")
                paths = sorted(glob(wild_card_path))
                if name not in best_runs:
                    best_runs[name] = {}
                if postfix not in best_runs[name]:
                    best_runs[name][postfix] = {}
                best_runs[name][postfix][env] = paths
    print(f"Best configs: {best_runs}")

    print(f"Saving best configs to {best_runs_path}")
    with open(best_runs_path, "w") as f:
        json.dump(best_runs, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="runs-optuna",
        help="Root directory of experiment runs.",
    )
    parser.add_argument(
        "--postfixes",
        type=str,
        default="all",
        help="Comma separated list of postfixes to filter runs by method.",
    )
    parser.add_argument(
        "--aggregate_fn",
        type=str,
        default="mean",
        choices=["mean", "median", "both"],
        help="Aggregation function to use for scores.",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use cached scores or not.",
    )
    args = parser.parse_args()
    if args.postfixes == "all":
        postfixes = get_all_postfixes(args.runs_dir)
    else:
        postfixes = args.postfixes.split(",")
    if args.aggregate_fn == "mean" or args.aggregate_fn == "both":
        figures_dir = "figures-hypsrch-mean"
        output_dir = os.path.join(args.runs_dir, figures_dir)
        main(
            args.runs_dir,
            output_dir,
            use_median=False,
            postfixes=postfixes,
            use_cache=args.cache,
        )
    if args.aggregate_fn == "median" or args.aggregate_fn == "both":
        figures_dir = "figures-hypsrch-median"
        output_dir = os.path.join(args.runs_dir, figures_dir)
        main(
            args.runs_dir,
            output_dir,
            use_median=True,
            postfixes=postfixes,
            use_cache=args.cache,
        )
