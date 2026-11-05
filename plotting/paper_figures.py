"""Figures 1, 2, 5, 6, 7, 10, 15, 16, the legends of Figures 1, 2, 6, 7, 11 and 16, and Appendix B.3 of the paper, written to figures/.

Usage: python plotting/paper_figures.py [fig1 fig2 fig5 fig6 fig7a fig7b fig7c fig10 fig15 fig16 legends held_out bounds]
"""

import json
import os
import re
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from common_utils import (
    SCORE_BOUNDS_PATH,
    MultiRunDataCache,
    RunDataCache,
    compute_iqm,
    get_avg_return_from_event_file,
    get_first_n_returns_from_event_file,
    normalize_method_scores,
)
from rliable import library as rly
from rliable import metrics
from tb_events_to_report import (
    aggregate_seeds,
    exponential_moving_average,
    load_seed_curve,
)

OUT = "figures"
TAB = plt.get_cmap("tab20").colors
BLUE, LIGHT_BLUE, ORANGE, LIGHT_ORANGE, GREEN, LIGHT_GREEN = TAB[:6]
TAB_RED, TAB_BROWN, TAB_PINK = TAB[6], TAB[10], TAB[12]
TAB_PURPLE = (0.5803921569, 0.4039215686, 0.7411764706)
CURVE_ORANGE = (0.9019607843, 0.5215686275, 0.0549019608)
DCPG_GREEN = (0.596, 0.875, 0.596)
ATARI4 = [f"{g}NoFrameskip-v4" for g in ("BeamRider", "Breakout", "Enduro", "Pong")]
ATARI7 = [
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
ATARI5 = [e for e in ATARI7 if not e.startswith(("Enduro", "Qbert"))]
SHORT = {"InvertedDoublePendulum-v4": "InD", "InvertedPendulum-v4": "InP"}
PPO, MIX, RIN = (
    "PPO ($r^\\mathrm{ex}$)",
    "PPO-LIRPG ($r^\\mathrm{ex} + r^\\mathrm{in}$)",
    "PPO-LIRPG ($r^\\mathrm{in}$)",
)

# lambda_in per game (Tables 4 and 5).
ATARI_MIXED_LAMBDA = {
    "BeamRider": 0.3,
    "Breakout": 0.01,
    "Enduro": 0.01,
    "Pong": 0.1,
    "Qbert": 0.01,
    "Seaquest": 0.1,
    "SpaceInvaders": 0.01,
}
ATARI_DCPG_LAMBDA = {
    "BeamRider": 0.1,
    "Breakout": 0.01,
    "Enduro": 0.9,
    "Pong": 0.99,
    "Qbert": 0.3,
    "Seaquest": 0.01,
    "SpaceInvaders": 0.9,
}
MUJOCO_MIXED_LAMBDA = {
    "HalfCheetah-v4": 0.99,
    "Hopper-v4": 0.99,
    "Walker2d-v4": 0.7,
    "InvertedPendulum-v4": 0.3,
    "InvertedDoublePendulum-v4": 0.1,
    "Swimmer-v4": 0.99,
    "Reacher-v4": 0.01,
}
TAGS = {
    0: "tuning_1_0_0",
    0.01: "tuning_0.99_.01_0",
    0.1: "tuning_0.9_.1_0",
    0.3: "tuning_0.7_.3_0",
    0.5: "tuning_0.5_.5_0",
    0.7: "tuning_0.3_.7_0",
    0.9: "tuning_0.1_.9_0",
    0.99: "tuning_0.01_.99_0",
    1: "tuning_0_1_0",
}
DCPG_REG = "tuning_0_1_0-ppg_vr_{}_pr_{}_tfreq_{}"


def game(env):
    return env.replace("NoFrameskip-v4", "")


def canvas(width, height, rects):
    """Figure of width x height points with axes at (x, y, w, h) point rectangles."""
    fig = plt.figure(figsize=(width / 72, height / 72))
    axes = [
        fig.add_axes([x / width, y / height, w / width, h / height])
        for x, y, w, h in rects
    ]
    return fig, axes


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    plt.close(fig)
    print(f"saved {OUT}/{name}.pdf", flush=True)


def pick(runs_dir, env, exp, seeds=(0, 1, 2)):
    """Event files of the runs in runs_dir on env with exp_name exp and one of the given seeds."""
    return sorted(
        f
        for s in seeds
        for f in glob(
            os.path.join(runs_dir, f"{env}__--{exp}--__{s}__*", "events.out.tfevents.*")
        )
    )


def random_floor(*runs_dirs):
    """{env: median final-100 score of the random-policy runs in runs_dirs}."""
    by_env = {}
    for d in runs_dirs:
        for f in glob(
            os.path.join(d, "*__--random_policy--__*", "events.out.tfevents.*")
        ):
            score = get_avg_return_from_event_file(f, window=100)
            if score is not None:
                by_env.setdefault(
                    os.path.basename(os.path.dirname(f)).split("__")[0], []
                ).append(score)
    return {env: float(np.median(v)) for env, v in by_env.items()}


def envs_with(runs_dir, exp):
    return sorted(
        {
            os.path.basename(d).split("__")[0]
            for d in glob(os.path.join(runs_dir, f"*__--{exp}--__*"))
        }
    )


def interval_estimates(paths, random_scores, cache_dir):
    """rliable mean with 95% bootstrap CI per method, using the smallest seed count of any game."""
    cache = MultiRunDataCache(cache_dir)
    normalized, _ = normalize_method_scores(
        paths, random_scores, map_names=False, cache=cache, window=100
    )
    cache.save()
    envs = list(paths)
    matrices = {}
    for label in paths[envs[0]]:
        per_env = [normalized[env][label] for env in envs]
        n = min(len(s) for s in per_env)
        matrices[label] = np.array([s[:n] for s in per_env]).T
    point, ci = rly.get_interval_estimates(
        matrices, lambda x: np.array([metrics.aggregate_mean(x)]), reps=50000
    )
    for label in matrices:
        print(
            f"  {label}: {point[label][0]:.3f} [{ci[label][0, 0]:.3f}, {ci[label][1, 0]:.3f}]",
            flush=True,
        )
    return point, ci, normalized


def bars(ax, point, ci, labels, colors, hatches, width, xlim):
    """Borderless bars spanning the CI, with a line at the point estimate."""
    for i, label in enumerate(labels):
        lower, upper = ci[label][:, 0]
        ax.bar(
            i,
            upper - lower,
            width=width,
            bottom=lower,
            color=colors[i],
            alpha=0.75,
            label=label,
            hatch=hatches[i] or None,
            edgecolor="black",
            linewidth=0,
        )
        ax.hlines(
            point[label][0],
            i - width / 2,
            i + width / 2,
            color="k",
            alpha=0.5,
            linewidth=1.5,
        )
    ax.set_xticks([])
    ax.set_xlim(*xlim)
    ax.grid(True, axis="y", alpha=0.25)


def aggregate_bars(
    name,
    W,
    H,
    rect,
    methods,
    paths,
    random_scores,
    cache_dir,
    width,
    xlim,
    tick_size,
    ylabel_size,
    legend=None,
    ymin=0.0,
):
    """One rliable mean-score bar per method; methods: [(label, colour, hatch)]."""
    print(name, flush=True)
    labels = [m[0] for m in methods]
    for env, by_method in paths.items():
        for label in labels:
            if len(by_method[label]) != 3:
                print(
                    f"  WARNING {env} {label}: {len(by_method[label])} runs", flush=True
                )
    point, ci, _ = interval_estimates(paths, random_scores, cache_dir)
    fig, (ax,) = canvas(W, H, [rect])
    bars(
        ax,
        point,
        ci,
        labels,
        [m[1] for m in methods],
        [m[2] for m in methods],
        width=width,
        xlim=xlim,
    )
    ax.set_ylim(ymin, 1.1)
    ax.set_yticks(np.round(np.arange(np.ceil(ymin / 0.2 - 1e-9) * 0.2, 1.0001, 0.2), 1))
    ax.tick_params(axis="y", labelsize=tick_size)
    if ylabel_size:
        ax.set_ylabel("Normalized Score", fontsize=ylabel_size)
    if legend:
        fig.legend(
            *ax.get_legend_handles_labels(), frameon=False, borderaxespad=0, **legend
        )
    save(fig, name)


def per_env_bars(
    name,
    W,
    H,
    rect,
    methods,
    paths,
    random_scores,
    cache_dir,
    width,
    offset,
    xlim,
    ylim,
    tick_shift,
):
    """Per-game mean +- population s.d. over seeds of each method's normalized score."""
    print(name, flush=True)
    envs = list(paths)
    cache = MultiRunDataCache(cache_dir)
    normalized, _ = normalize_method_scores(
        paths, random_scores, map_names=False, cache=cache, window=100
    )
    cache.save()
    fig, (ax,) = canvas(W, H, [rect])
    n = len(methods)
    for i, (label, color, hatch) in enumerate(methods):
        for g, env in enumerate(envs):
            v = np.array(normalized[env][label])
            x = g + (i - (n - 1) / 2) * offset
            ax.bar(
                x,
                2 * v.std(),
                width=width,
                bottom=v.mean() - v.std(),
                color=color,
                alpha=0.7,
                hatch="////" if hatch else None,
                edgecolor="black",
                linewidth=0,
            )
            ax.hlines(v.mean(), x - width / 3, x + width / 3, color="k", linewidth=2)
            print(f"  {env} {label}: {v.mean():.3f} +- {v.std():.3f}", flush=True)
    ax.set_xticks(np.arange(len(envs)) + tick_shift)
    ax.set_xticklabels(
        [SHORT.get(e, e[:3]) for e in envs],
        rotation=30,
        ha="right",
        fontsize=23,
        rotation_mode="anchor",
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylabel("Normalized Score", fontsize=23)
    ax.grid(True, axis="y", alpha=0.25)
    save(fig, name)


def comparison_paths(domain):
    """PPO, mixed reward at the Table 4/5 lambda_in, and intrinsic-only runs of lambda_sweep.sh."""
    d = f"runs-{domain}-tuning"
    if domain == "atari":
        envs, lam = ATARI7, {e: ATARI_MIXED_LAMBDA[game(e)] for e in ATARI7}
    else:
        envs, lam = sorted(MUJOCO_MIXED_LAMBDA), MUJOCO_MIXED_LAMBDA
    return {
        env: {
            PPO: pick(d, env, TAGS[0]),
            MIX: pick(d, env, TAGS[lam[env]]),
            RIN: pick(d, env, TAGS[1]),
        }
        for env in envs
    }


THREE = [(PPO, BLUE, ""), (MIX, ORANGE, ""), (RIN, GREEN, "")]


def fig1():
    aggregate_bars(
        "mujoco_overall",
        143.31876,
        203.48062,
        (52.41875, 14.840625, 83.7, 181.44),
        THREE,
        comparison_paths("mujoco"),
        random_floor("runs-mujoco-tuning"),
        "runs-mujoco-tuning",
        width=0.7273,
        xlim=(-0.5, 2.5),
        tick_size=13,
        ylabel_size=14.4,
    )
    aggregate_bars(
        "atari_overall",
        125.77187,
        203.48062,
        (34.871875, 14.840625, 83.7, 181.44),
        THREE,
        comparison_paths("atari"),
        random_floor("runs-atari-tuning"),
        "runs-atari-tuning",
        width=0.7273,
        xlim=(-0.5, 2.5),
        tick_size=13,
        ylabel_size=None,
    )


def fig2():
    per_env_bars(
        "mujoco_per_env_for_combined",
        828.565,
        252.69376,
        (69.075, 56.4534402342, 752.29, 189.0403129875),
        THREE,
        comparison_paths("mujoco"),
        random_floor("runs-mujoco-tuning"),
        "runs-mujoco-tuning",
        width=0.15,
        offset=0.2,
        xlim=(-0.6025, 6.6025),
        ylim=(-0.0235, 1.0818),
        tick_shift=-0.0067,
    )
    per_env_bars(
        "atari_per_env_for_combined",
        828.565,
        252.64406,
        (69.075, 56.8987527342, 752.29, 188.5453129875),
        THREE,
        comparison_paths("atari"),
        random_floor("runs-atari-tuning"),
        "runs-atari-tuning",
        width=0.15,
        offset=0.2,
        xlim=(-0.6025, 6.6025),
        ylim=(-0.0141, 1.1423),
        tick_shift=-0.0067,
    )


def fig5_paths():
    d = "runs-atari-comparison-unshared"
    groups = [
        (PPO, lambda e: TAGS[0], lambda e: TAGS[0], BLUE, LIGHT_BLUE),
        (
            MIX,
            lambda e: TAGS[ATARI_MIXED_LAMBDA[game(e)]],
            lambda e: TAGS[0.01],
            ORANGE,
            LIGHT_ORANGE,
        ),
        (RIN, lambda e: TAGS[1], lambda e: TAGS[1], GREEN, LIGHT_GREEN),
    ]
    methods, paths = [], {env: {} for env in ATARI7}
    for name, shared_tag, unshared_tag, dark, light in groups:
        for sharing, tag, color, hatch in (
            ("shared", shared_tag, dark, ""),
            ("non-shared", unshared_tag, light, "o"),
        ):
            label = f"{name} ({sharing})"
            methods.append((label, color, hatch))
            for env in ATARI7:
                paths[env][label] = pick(
                    d, env, f"{tag(env)}-{sharing.replace('non-', 'un')}"
                )
    return methods, paths


def fig5():
    methods, paths = fig5_paths()
    print("param_sharing_atari", flush=True)
    point, ci, _ = interval_estimates(
        paths, random_floor("runs-atari-tuning"), "runs-atari-comparison-unshared"
    )
    fig, (ax,) = canvas(517.4775, 205.51187, [(54.0125, 13.075, 195.3, 181.44)])
    bars(ax, point, ci, *zip(*methods), width=0.7692, xlim=(-0.6732, 5.6732))
    for x in (-0.6155, 1.4613, 3.5380):
        ax.axvline(x, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("Normalized Score", fontsize=14.4)
    ax.legend(
        loc="center left", bbox_to_anchor=(1.0507, 0.5), fontsize=12, frameon=True
    )
    save(fig, "param_sharing_atari")


FIG6 = [
    (PPO, BLUE, ""),
    (f"{PPO} (DCPG)", LIGHT_BLUE, "//"),
    (MIX, ORANGE, ""),
    (f"{MIX} (DCPG)", LIGHT_ORANGE, "//"),
    (RIN, GREEN, ""),
    (f"{RIN} (DCPG)", LIGHT_GREEN, "//"),
]


def fig6_paths():
    """Figure 6 runs, seeds 0-2."""
    base, dcpg = "runs-atari-tuning", "runs-atari-tuning-dcpg"
    return {
        env: {
            PPO: pick(base, env, TAGS[0]),
            f"{PPO} (DCPG)": pick(dcpg, env, f"{TAGS[0]}-dcpg"),
            MIX: pick(base, env, TAGS[ATARI_MIXED_LAMBDA[game(env)]]),
            f"{MIX} (DCPG)": pick(
                dcpg, env, f"{TAGS[ATARI_DCPG_LAMBDA[game(env)]]}-dcpg"
            ),
            RIN: pick(base, env, TAGS[1]),
            f"{RIN} (DCPG)": pick(
                "runs-atari-dcpg-reg", env, DCPG_REG.format("0.0", "1.0", 32)
            ),
        }
        for env in ATARI7
    }


def fig6():
    paths, floor = fig6_paths(), random_floor("runs-atari-tuning")
    per_env_bars(
        "atari_ppg_per_env_noleg",
        828.565,
        252.64406,
        (69.075, 56.8987527342, 752.29, 188.5453129875),
        FIG6,
        paths,
        floor,
        "runs-atari-tuning",
        width=0.1,
        offset=0.13333,
        xlim=(-0.7216, 6.7216),
        ylim=(-0.0141, 1.1423),
        tick_shift=-0.1098,
    )
    aggregate_bars(
        "atari_ppg_overall_noleg",
        267.46565,
        205.24625,
        (64.965625, 16.60625, 195.3, 181.44),
        FIG6,
        paths,
        floor,
        "runs-atari-tuning",
        width=0.7692,
        xlim=(-0.6731, 5.6731),
        tick_size=16,
        ylabel_size=22,
    )


def held_out_paths():
    """Held-out seeds 10-12 of experiments/dcpg.sh."""
    d, seeds = "runs-atari-heldout-mixed", (10, 11, 12)
    return {
        env: {
            MIX: pick(d, env, TAGS[ATARI_MIXED_LAMBDA[game(env)]], seeds),
            f"{MIX} (DCPG)": pick(d, env, f"{TAGS[0.5]}-dcpg", seeds),
        }
        for env in ATARI7
    }


def held_out():
    """Appendix B.3: aggregate with 95% CI, then per-game mean +- s.d."""
    paths = held_out_paths()
    print("held_out", flush=True)
    _, _, normalized = interval_estimates(
        paths, random_floor("runs-atari-tuning"), "runs-atari-heldout-mixed"
    )
    for env in ATARI7:
        print(
            f"  {env:28s} "
            + "  ".join(
                f"{m}: {np.mean(normalized[env][m]):.3f} +- {np.std(normalized[env][m]):.3f}"
                for m in paths[env]
            ),
            flush=True,
        )


DCPG_THREE = [(PPO, BLUE, ""), (RIN, GREEN, ""), (f"{RIN} (DCPG)", LIGHT_GREEN, "//")]


def suite_paths(runs_dir):
    """PPO, intrinsic-only and intrinsic-only + DCPG runs of atari49.sh or procgen.sh."""
    return {
        env: {
            PPO: pick(runs_dir, env, TAGS[0]),
            RIN: pick(runs_dir, env, TAGS[1]),
            f"{RIN} (DCPG)": pick(runs_dir, env, f"{TAGS[1]}-dcpg"),
        }
        for env in envs_with(runs_dir, f"{TAGS[1]}-dcpg")
    }


def fig7a():
    aggregate_bars(
        "rliable_atari49_aggregate_nolegend",
        143.31876,
        203.48062,
        (52.41875, 14.840625, 83.7, 181.44),
        DCPG_THREE,
        suite_paths("runs-atari-complete"),
        random_floor("runs-atari-complete"),
        "runs-atari-complete",
        width=0.7273,
        xlim=(-0.5, 2.5),
        tick_size=13,
        ylabel_size=14.4,
    )


def fig7b():
    # ProcGen floor: IQM of the first 10 episodes of the plotted runs.
    aggregate_bars(
        "rliable_procgen5_aggregate_nolegend",
        154.20938,
        199.34,
        (63.309375, 10.7, 83.7, 181.44),
        DCPG_THREE,
        suite_paths("runs-procgen"),
        {},
        "runs-procgen",
        width=0.7273,
        xlim=(-0.5, 2.5),
        tick_size=13,
        ylabel_size=14.4,
        ymin=-0.2898,
    )


def fig7c_paths():
    d = "runs-atari-practice"
    return {
        env: {
            RIN: pick(d, env, "practice"),
            f"{RIN} (DCPG)": pick(d, env, "practice_dcpg"),
        }
        for env in ATARI4
    }


def fig7c():
    print("rliable_practice4_aggregate_nolegend", flush=True)
    paths = fig7c_paths()
    point, ci, _ = interval_estimates(
        paths, random_floor("runs-atari-tuning"), "runs-atari-practice"
    )
    fig, (ax,) = canvas(142.70938, 202.9025, [(51.809375, 14.2625, 83.7, 181.44)])
    bars(
        ax,
        point,
        ci,
        list(paths[ATARI4[0]]),
        [GREEN, DCPG_GREEN],
        ["", "//"],
        width=0.4286,
        xlim=(-0.2857, 1.2857),
    )
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Normalized Score", fontsize=15)
    save(fig, "rliable_practice4_aggregate_nolegend")


FIG10 = [
    (PPO, BLUE, ""),
    (RIN, GREEN, ""),
    (f"{RIN} (decoupling)", TAB_RED, ""),
    (f"{RIN} (decoupling + policy reg)", LIGHT_GREEN, "//"),
    (f"{RIN} (decoupling + value reg)", TAB_BROWN, ""),
    (f"{RIN} (decoupling + policy and value reg)", TAB_PINK, ""),
]


def fig10_paths():
    reg = "runs-atari-dcpg-reg"
    exps = [("runs-atari-tuning", TAGS[0]), ("runs-atari-tuning", TAGS[1])] + [
        (reg, DCPG_REG.format(bv, bp, 32))
        for bv, bp in (("0.0", "0.0"), ("0.0", "1.0"), ("1.0", "0.0"), ("1.0", "1.0"))
    ]
    return {
        env: {m[0]: pick(d, env, e) for m, (d, e) in zip(FIG10, exps)} for env in ATARI7
    }


def fig10():
    aggregate_bars(
        "reg_ablation_atari",
        793.0786,
        205.24625,
        (64.965625, 16.60625, 195.3, 181.44),
        FIG10,
        fig10_paths(),
        random_floor("runs-atari-tuning"),
        "runs-atari-dcpg-reg",
        width=0.7692,
        xlim=(-0.6731, 5.6731),
        tick_size=16,
        ylabel_size=22,
        legend=dict(
            loc="upper left",
            bbox_to_anchor=(278.7 / 793.0786, 193.85 / 205.24625),
            fontsize=17.28,
        ),
    )


def fig15():
    d = "runs-atari-practice"
    W, H = 737.424, 174.54262
    fig, axes = canvas(
        W,
        H,
        [(x, 73.072, 134.2775, 76.88) for x in (64.114, 241.3915, 418.669, 595.9465)],
    )
    cache = RunDataCache(os.path.join(d, ".cache"))
    methods = [
        (RIN, "practice", CURVE_ORANGE),
        (f"{RIN} (DCPG)", "practice_dcpg", GREEN),
    ]
    for ax, env in zip(axes, ATARI4):
        for label, exp, color in methods:
            for i, p in enumerate(pick(d, env, exp)):
                steps, values = load_seed_curve(p, "charts/episodic_return", cache)
                ax.plot(
                    steps,
                    exponential_moving_average(values, 0.99),
                    color=color,
                    linewidth=0.8,
                    label=label if i == 0 else None,
                )
        ax.set_title(game(env), fontsize=15)
        ax.set_xticks([2e6, 4e6, 6e6, 8e6])
        ax.tick_params(labelsize=12)
        ax.xaxis.get_offset_text().set_fontsize(10)
        ax.grid(True, alpha=0.3)
    cache.save()
    fig.text(
        (W / 2 - 2.1) / W, 9.9 / H, "Steps", ha="center", va="baseline", fontsize=13
    )
    fig.text(
        13.5 / W,
        91.9 / H,
        "Episodic Return",
        rotation=90,
        ha="center",
        va="center",
        fontsize=13,
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(503.8 / W, 65.4 / H),
        borderaxespad=0,
        fontsize=15,
    )
    save(fig, "per_env_practice4")


def fig16():
    d = "runs-atari-practice"
    cache = RunDataCache(os.path.join(d, ".cache"))
    for env in ATARI4:
        fig, (ax,) = canvas(264.79814, 193.9375, [(55.028125, 50.1375, 202.57, 136.6)])
        for tag, color in (
            ("losses/inner_v_loss", TAB_RED),
            ("losses/inner_pg_loss", TAB_PURPLE),
            ("losses/inner_joint_loss", "k"),
        ):
            common, mean, std, _ = aggregate_seeds(
                pick(d, env, "practice"), tag, cache, 0.9, 500
            )
            ax.plot(common, mean, color=color, linewidth=2.2)
            ax.fill_between(
                common, mean - std, mean + std, color=color, alpha=0.2, linewidth=0
            )
        ax.tick_params(labelsize=16)
        ax.xaxis.get_offset_text().set_fontsize(17)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0)
        ax.set_xlabel("Step", fontsize=17)
        ax.set_ylabel("Value", fontsize=17)
        save(fig, f"losses-prac_{game(env)}")
    cache.save()


def legend_page(name, W, H, corner, handles, labels, **kw):
    """Page of W x H points holding only a legend whose box has its lower-left corner at corner (points)."""
    fig = plt.figure(figsize=(W / 72, H / 72))
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(corner[0] / W, corner[1] / H),
        borderaxespad=0,
        **kw,
    )
    save(fig, name)


def legends():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    def patches(methods, hatch="////", **kw):
        return [
            Patch(facecolor=c, hatch=hatch if h else None, linewidth=0, **kw)
            for _, c, h in methods
        ]

    def italic(label):
        return label.replace("\\mathrm", "")

    # The first two pages are the crops the paper takes of a larger plot.
    legend_page(
        "per_env_legend_for_combined",
        696.133,
        27.8564,
        (-4.620426, -6.57),
        patches(THREE, alpha=0.7, edgecolor="black"),
        [m[0] for m in THREE],
        ncol=3,
        fontsize=20,
    )
    # Figure 1: two-row legend at printed size (7.5pt), included without scaling.
    fig = plt.figure(figsize=(4, 1))
    order = [
        THREE[0],
        THREE[2],
        THREE[1],
    ]  # column-major fill reads "PPO, mixed" then "intrinsic-only"
    fig.legend(
        patches(order, alpha=0.7, edgecolor="black"),
        [m[0] for m in order],
        loc="center",
        ncol=2,
        fontsize=7.5,
        frameon=False,
        handlelength=1.3,
        handletextpad=0.4,
        columnspacing=0.9,
        labelspacing=0.3,
        borderaxespad=0,
    )
    fig.savefig(
        os.path.join(OUT, "overall_legend_2row.pdf"),
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close(fig)
    legend_page(
        "atari_ppg_legend",
        810.479,
        69.1592,
        (3.083048, 2.49),
        patches(FIG6, alpha=0.7, edgecolor="black"),
        ["PPO", "PPO (DCPG)"] + [italic(m[0]) for m in FIG6[2:]],
        ncol=3,
        fontsize=18,
    )
    legend_page(
        "legend_dcpg_2col",
        397.60075,
        59.964,
        (0.050377, 4.232),
        patches(DCPG_THREE, hatch="//", edgecolor="none"),
        [italic(m[0]) for m in DCPG_THREE],
        ncol=2,
        fontsize=15,
        frameon=False,
        handletextpad=0.6,
        columnspacing=1.5,
    )
    lines = [
        Line2D([], [], color=c, linewidth=2.5) for c in (TAB_RED, TAB_PURPLE, "#1b1b1b")
    ]
    legend_page(
        "legend_losses",
        132,
        70.2,
        (0.070312, -0.1734375),
        lines,
        ["Value Loss", "Policy Loss", "Joint Loss"],
        fontsize=15,
        frameon=False,
        handletextpad=0.6,
    )
    legend_page(
        "legend_losses_h",
        565.0752,
        29.28,
        (0.646965, 0.64),
        lines,
        ["Value Loss", "Policy Loss", "Policy Loss + 0.5 $\\times$ Value Loss"],
        ncol=3,
        fontsize=15,
        frameon=False,
        handletextpad=0.6,
        columnspacing=1.5,
    )


def sweep_paths(runs_dir, exp):
    """{env: {exp_name: event files}} for seed 0-2 runs whose exp_name fully matches exp."""
    paths = {}
    for r in sorted(glob(os.path.join(runs_dir, "*__*__[012]__*"))):
        env, name = os.path.basename(r).split("__")[:2]
        if re.fullmatch(f"--{exp}--", name):
            paths.setdefault(env, {}).setdefault(name, []).extend(
                glob(os.path.join(r, "events.out.tfevents.*"))
            )
    return paths


def table_paths():
    """Rows of Tables 10-12 (plotting/paper_tables.py)."""
    reg = "runs-atari-dcpg-reg"
    t10 = {
        env: {
            "shared": pick("runs-atari-tuning", env, TAGS[1]),
            "pretrained": pick(
                "runs-atari-comparison-pretrained_init", env, "pretrained_policy_only"
            ),
            "dcpg": pick(reg, env, DCPG_REG.format("0.0", "1.0", 32)),
        }
        for env in ATARI7
    }
    t11 = {
        env: {
            f"t{t}": pick(
                reg if t == 32 else "runs-atari-dcpg-tfreq",
                env,
                DCPG_REG.format("0.0", "1.0", t),
            )
            for t in (1, 8, 16, 32, 64)
        }
        for env in ATARI5
    }
    t12 = {
        env: {
            v: pick("runs-atari-vf_coef_sweep", env, f"{TAGS[1]}-vf_{v}")
            for v in ("0.01", "0.05", "0.1", "0.25", "0.5")
        }
        for env in ATARI5
    }
    return {"Table 10": t10, "Table 11": t11, "Table 12": t12}


def scored_paths():
    """{source: {env: {method: event files}}} for every normalized score the paper reports."""
    import plot_alt_strategies_figure

    sources = {
        "Fig 1/2 MuJoCo": lambda: comparison_paths("mujoco"),
        "Fig 1/2 Atari": lambda: comparison_paths("atari"),
        "Fig 3 MuJoCo": lambda: sweep_paths(
            "runs-mujoco-tuning", r"tuning_[0-9.]+_[0-9.]+_0"
        ),
        "Fig 3 Atari": lambda: sweep_paths(
            "runs-atari-tuning", r"tuning_[0-9.]+_[0-9.]+_0"
        ),
        "Fig 5": lambda: fig5_paths()[1],
        "Fig 6": fig6_paths,
        "Appendix B.3": held_out_paths,
        "Fig 8": lambda: sweep_paths(
            "runs-atari-tuning-dcpg", r"tuning_[0-9.]+_[0-9.]+_0-dcpg"
        ),
        "Fig 7a": lambda: suite_paths("runs-atari-complete"),
        "Fig 7b": lambda: suite_paths("runs-procgen"),
        "Fig 7c": fig7c_paths,
        "Fig 9": plot_alt_strategies_figure.build_game_method_paths,
        "Fig 10": fig10_paths,
        "Tables": table_paths,
    }
    out = {}
    for source, build in sources.items():
        try:
            built = build()
        except Exception as e:
            print(f"[WARNING] {source}: skipped ({type(e).__name__}: {e})")
            continue
        out.update(built if source == "Tables" else {source: built})
    return out


def bounds():
    """Update plotting/score_bounds.json with each game's floor and ceiling; games without runs keep their entry."""
    cache = MultiRunDataCache("runs-atari-tuning")
    best = {}
    for by_env in scored_paths().values():
        for env, by_method in by_env.items():
            for files in by_method.values():
                for f in files:
                    score = get_avg_return_from_event_file(f, window=100, cache=cache)
                    if score is not None and score > best.get(env, float("-inf")):
                        best[env] = score
    cache.save()
    # The 7 analysis games take their floor from runs-atari-tuning, the other Atari games from runs-atari-complete.
    floors = {}
    for d in ("runs-atari-complete", "runs-atari-tuning", "runs-mujoco-tuning"):
        floors.update(random_floor(d))
    # ProcGen has no random-policy runs: interquartile mean of the first 10 episodes of the Figure 7(b) runs.
    for env, by_method in suite_paths("runs-procgen").items():
        first = [
            r
            for files in by_method.values()
            for f in files
            if get_avg_return_from_event_file(f, window=100, cache=cache) is not None
            for r in get_first_n_returns_from_event_file(f, n=10, cache=cache) or []
        ]
        if first:
            floors[env] = float(compute_iqm(first))
    cache.save()
    merged = (
        json.load(open(SCORE_BOUNDS_PATH)) if os.path.exists(SCORE_BOUNDS_PATH) else {}
    )
    for env, ceiling in best.items():
        entry = merged.setdefault(env, {})
        entry["max"] = max(ceiling, entry.get("max", float("-inf")))
    for env, floor in floors.items():
        merged.setdefault(env, {})["min"] = floor
    for env, b in sorted(merged.items()):
        print(
            f"{env:32s} {b.get('min', float('nan')):12.2f} {b['max']:12.2f}", flush=True
        )
    with open(SCORE_BOUNDS_PATH, "w") as fh:
        json.dump(merged, fh, indent=1)
    print(f"wrote {SCORE_BOUNDS_PATH}")


if __name__ == "__main__":
    for name in sys.argv[1:] or [
        "fig1",
        "fig2",
        "fig5",
        "fig6",
        "fig7a",
        "fig7b",
        "fig7c",
        "fig10",
        "fig15",
        "fig16",
        "legends",
        "held_out",
    ]:
        globals()[name]()
