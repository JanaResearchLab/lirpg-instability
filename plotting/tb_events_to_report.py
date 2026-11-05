"""Plot TensorBoard scalars per environment, grouped by setting and run postfix.

Usage: python tb_events_to_report.py --runs_dir runs-atari-rnd --settings PPO-RND --tags charts/mean_r_rnd
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
    RunDataCache,
    extract_config_from_method,
    extract_method_name,
    get_event_data_from_event_file,
    get_setting_from_config,
    method_name_mapping,
)

plt.rcParams.update({"font.size": 15})

tag_label_mapping = {
    "losses/inner_v_loss": "value",
    "losses/inner_pg_loss": "policy",
    "losses/inner_joint_loss": "joint",
    "losses/inner_entropy_loss": "entropy",
    "losses/inner_grad_v": "grad value",
    "losses/inner_grad_p": "grad policy",
    "losses/outer_v_loss": "value",
    "losses/outer_pg_loss": "policy",
    "losses/outer_joint_loss": "joint",
    "charts/mean_r_rnd": r"mean $r^{\mathrm{rnd}}$",
    "charts/episodic_return": "episodic return",
}

tag_color_mapping = {
    "losses/inner_v_loss": "#d62728",
    "losses/inner_pg_loss": "#9467bd",
    "losses/inner_joint_loss": "#1b1b1b",
    "losses/outer_v_loss": "#d62728",
    "losses/outer_pg_loss": "#9467bd",
    "losses/outer_joint_loss": "#1b1b1b",
}

# Fallbacks for tags missing from event files: {tag: (combine_fn, source_tags)}.
synthetic_tag_recipes = {
    "losses/inner_joint_loss": (
        lambda vals: vals[0] + 0.5 * vals[1],
        ["losses/inner_pg_loss", "losses/inner_v_loss"],
    ),
    "losses/outer_joint_loss": (
        lambda vals: vals[0] + 0.5 * vals[1],
        ["losses/outer_pg_loss", "losses/outer_v_loss"],
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot TF scalars across envs, settings, and method postfixes."
    )
    parser.add_argument("--runs_dir", type=str, required=True)
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Envs to plot, one subplot each (default: all in --runs_dir).",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=None,
        help="Settings to keep, as named by get_setting_from_config (default: all).",
    )
    parser.add_argument(
        "--postfixes",
        type=str,
        default=None,
        help="Comma-separated postfixes; an empty entry means no postfix (default: all).",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        required=True,
        help="TF event tags to plot (one line per tag, per group).",
    )
    parser.add_argument("--log_scale", action="store_true")
    parser.add_argument(
        "--combine_envs",
        action="store_true",
        help="Plot all envs on one shared axes.",
    )
    parser.add_argument(
        "--no_legend",
        action="store_true",
        help="Hide the legend (see generate_legend_pdf.py).",
    )
    parser.add_argument("--smoothing", type=float, default=0.9)
    parser.add_argument("--n_cols", type=int, default=3)
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    return parser.parse_args()


def exponential_moving_average(data, smoothing_factor=0.9):
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return data
    alpha = 1.0 - smoothing_factor
    smoothed = np.empty_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def load_curve(path, cache, metric):
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


def interpolate_to_common(steps_list, values_list, num_points=500):
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


def split_method(method):
    """Return (config_str, postfix); postfix keeps its leading '-' and is '' if absent."""
    if "-" in method:
        config_str, post = method.split("-", 1)
        return config_str, f"-{post}"
    return method, ""


def discover_envs(runs_dir):
    """Return env names parsed from run folder names."""
    envs = set()
    for folder in os.listdir(runs_dir):
        if not os.path.isdir(os.path.join(runs_dir, folder)):
            continue
        m = re.match(r"(.+?)__--", folder)
        if m:
            envs.add(m.group(1))
    return sorted(envs)


def discover_runs(runs_dir, env, postfix_filter, setting_filter):
    """Return {(setting, postfix): [event_paths]} for a single env."""
    pattern = os.path.join(runs_dir, f"{env}__--*--__*__*", "events.out.tfevents.*")
    grouped = defaultdict(list)
    for path in sorted(glob(pattern)):
        method = extract_method_name(path)
        _, postfix = split_method(method)
        config = extract_config_from_method(method)
        if config is not None:
            setting = get_setting_from_config(*config)
        else:
            # Non-tuning runs (e.g. practice-match) are labelled by their exp_name.
            setting, postfix = method_name_mapping.get(method), ""
        if setting is None:
            continue
        if postfix_filter is not None and postfix not in postfix_filter:
            continue
        if setting_filter is not None and setting not in setting_filter:
            continue
        grouped[(setting, postfix)].append(path)
    return grouped


def tag_short(tag):
    return tag_label_mapping.get(tag, tag.split("/")[-1])


def build_label(
    env,
    setting,
    postfix,
    tag,
    multi_env,
    multi_setting,
    multi_postfix,
    multi_tag,
):
    env = env.replace("NoFrameskip-v4", "")
    parts = []
    if multi_env:
        parts.append(env)
    if multi_setting:
        parts.append(setting)
    if multi_postfix:
        parts.append(postfix or "no postfix")
    if multi_tag:
        parts.append(tag_short(tag))
    if not parts:
        return setting or (postfix or "no postfix")
    return " — ".join(parts)


def load_seed_curve(path, tag, cache):
    """Return (steps, values) for one run, built from synthetic_tag_recipes if the tag is missing."""
    steps, values = load_curve(path, cache, tag)
    if steps is not None and len(steps) >= 2:
        return steps, values

    recipe = synthetic_tag_recipes.get(tag)
    if recipe is None:
        return None, None
    combine, source_tags = recipe

    src_steps_list, src_values_list = [], []
    for src in source_tags:
        s, v = load_curve(path, cache, src)
        if s is None or len(s) < 2:
            return None, None
        src_steps_list.append(s)
        src_values_list.append(v)

    common, interp = interpolate_to_common(src_steps_list, src_values_list)
    if len(common) == 0:
        return None, None
    combined = combine(list(interp))
    if cache is not None:
        cache.cache_data(
            path,
            list(common),
            list(combined),
            list(range(1, len(combined) + 1)),
            metric=tag,
        )
    return common, combined


def aggregate_seeds(paths, tag, cache, smoothing, num_points):
    seeds = []
    for path in paths:
        steps, values = load_seed_curve(path, tag, cache)
        if steps is None or len(steps) < 2:
            continue
        seeds.append((steps, values))
    if not seeds:
        return None
    steps_list = [s for s, _ in seeds]
    values_list = [v for _, v in seeds]
    common, interp = interpolate_to_common(
        steps_list, values_list, num_points=num_points
    )
    if len(common) == 0:
        return None
    smoothed = np.array([exponential_moving_average(v, smoothing) for v in interp])
    return common, smoothed.mean(axis=0), smoothed.std(axis=0), len(seeds)


def main():
    args = parse_args()

    postfix_filter = (
        [p for p in args.postfixes.split(",")] if args.postfixes is not None else None
    )
    setting_filter = list(args.settings) if args.settings is not None else None
    tags = list(args.tags)
    envs = list(args.envs) if args.envs is not None else discover_envs(args.runs_dir)
    if not envs:
        raise SystemExit(f"No environments found under {args.runs_dir}.")

    cache_dir = args.cache_dir or os.path.join(args.runs_dir, ".cache")
    cache = RunDataCache(cache_dir)

    grouped_per_env = {}
    for env in envs:
        groups = discover_runs(args.runs_dir, env, postfix_filter, setting_filter)
        if not groups:
            print(f"[skip] no matching runs for env={env}")
            continue
        grouped_per_env[env] = groups
    envs = list(grouped_per_env.keys())
    if not envs:
        raise SystemExit("No envs have runs matching the given filters.")

    settings_seen, postfixes_seen = set(), set()
    for groups in grouped_per_env.values():
        for setting, postfix in groups:
            settings_seen.add(setting)
            postfixes_seen.add(postfix)

    multi_setting = len(settings_seen) > 1
    multi_postfix = len(postfixes_seen) > 1
    multi_tag = len(tags) > 1

    multi_env = args.combine_envs and len(envs) > 1

    cmap = plt.get_cmap("tab10")
    sorted_settings = sorted(settings_seen)
    sorted_postfixes = sorted(postfixes_seen)
    color_lookup = {}
    idx = 0
    if args.combine_envs:
        for env in envs:
            for s in sorted_settings:
                for p in sorted_postfixes:
                    for t in tags:
                        color_lookup[(env, s, p, t)] = cmap(idx % 10)
                        idx += 1
    else:
        for s in sorted_settings:
            for p in sorted_postfixes:
                for t in tags:
                    if t in tag_color_mapping:
                        color_lookup[(s, p, t)] = tag_color_mapping[t]
                    else:
                        color_lookup[(s, p, t)] = cmap(idx % 10)
                        idx += 1

    n = len(envs)
    if args.combine_envs:
        fig, ax_single = plt.subplots(figsize=(7, 6))
        axes = None
        n_cols = 1
        n_rows = 1
    else:
        n_cols = min(args.n_cols, n)
        n_rows = math.ceil(n / n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.0 * n_cols, 3.0 * n_rows),
            squeeze=False,
        )

    legend_handles = {}

    for ei, env in enumerate(envs):
        ax = ax_single if args.combine_envs else axes[ei // n_cols][ei % n_cols]
        groups = grouped_per_env[env]
        for (setting, postfix), paths in sorted(groups.items()):
            for tag in tags:
                agg = aggregate_seeds(
                    paths, tag, cache, args.smoothing, args.num_points
                )
                if agg is None:
                    print(
                        f"[skip] no seeds for env={env} setting={setting} "
                        f"postfix={postfix or '<empty>'} tag={tag}"
                    )
                    continue
                common, mean, std, _ = agg
                color = (
                    color_lookup[(env, setting, postfix, tag)]
                    if args.combine_envs
                    else color_lookup[(setting, postfix, tag)]
                )
                label = build_label(
                    env,
                    setting,
                    postfix,
                    tag,
                    multi_env,
                    multi_setting,
                    multi_postfix,
                    multi_tag,
                )
                (line,) = ax.plot(common, mean, color=color, linewidth=1.4, label=label)
                ax.fill_between(
                    common,
                    mean - std,
                    mean + std,
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                )
                legend_handles.setdefault(label, line)

        if not args.combine_envs:
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, alpha=0.25)
            ax.margins(x=0)
            if args.log_scale:
                ax.set_yscale("log")

    if args.combine_envs:
        ax_single.tick_params(axis="both")
        ax_single.grid(True, alpha=0.25)
        ax_single.margins(x=0)
        if args.log_scale:
            ax_single.set_yscale("log")
    else:
        for j in range(n, n_rows * n_cols):
            axes[j // n_cols][j % n_cols].axis("off")

    cache.save()

    if (
        not args.no_legend
        and (multi_env or multi_setting or multi_postfix or multi_tag)
        and legend_handles
    ):
        fig.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, len(legend_handles)),
            framealpha=0.9,
        )

    ylabel = tag_short(tags[0]) if len(tags) == 1 else "Value"
    if args.log_scale:
        ylabel += " (log scale)"

    single_axes = args.combine_envs or n == 1
    if single_axes:
        target_ax = ax_single if args.combine_envs else axes[0][0]
        target_ax.set_xlabel("Step")
        target_ax.set_ylabel(ylabel)
    else:
        fig.supxlabel("Step")
        fig.supylabel(ylabel)
    plt.tight_layout()

    figures_dir = os.path.join(args.runs_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    tag_slug = "_".join(t.replace("/", "_") for t in tags)
    output = args.output or os.path.join(figures_dir, f"tb_plot_{tag_slug}.png")
    plt.savefig(output, bbox_inches="tight", dpi=200)
    pdf_path = output.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output} and {pdf_path}")


if __name__ == "__main__":
    main()
