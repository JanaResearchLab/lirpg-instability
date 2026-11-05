"""Figure 17: still frames of a run's per-action r_in at a few indicative states.

Usage: python plotting/plot_r_in_states.py --lirpg_run_dir <run> --dcpg_run_dir <run> [--survey_run_dirs <run> ...]
"""

import argparse
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import AtariAgent
from intrinsic_reward_module import IntrinsicRewardFn
from plotting.visualize_r_in import make_env
from utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Single-threaded for reproducible rollouts.
torch.set_num_threads(1)


# game -> [(selector, lookahead in agent steps, label)]
SELECTOR_TABLE = {
    "Pong": [
        ("pre_reward", 6, "about to return the ball / score"),
        ("pre_setback", 6, "about to concede a point"),
        ("argmax_diverse", 0, "most action-dependent state"),
    ],
    "Breakout": [
        ("pre_reward", 4, "ball about to break a brick"),
        ("pre_setback", 6, "ball about to be missed"),
        ("argmax_diverse", 0, "most action-dependent state"),
    ],
    "BeamRider": [
        ("pre_reward", 6, "enemy ship about to be destroyed"),
        ("pre_setback", 6, "about to be destroyed"),
        ("argmax_diverse", 0, "most action-dependent state"),
    ],
    "Seaquest": [
        ("pre_reward", 6, "about to shoot an enemy / collect a diver"),
        ("pre_setback", 8, "about to run out of oxygen / be hit"),
        ("argmax_diverse", 0, "most action-dependent state"),
    ],
    "SpaceInvaders": [
        ("pre_reward", 6, "invader about to be hit"),
        ("pre_setback", 6, "about to be hit by a bomb"),
        ("argmax_diverse", 0, "most action-dependent state"),
    ],
    "Qbert": [
        ("pre_reward", 4, "cube about to be flipped"),
        ("pre_setback", 6, "about to be caught / fall off"),
        ("argmax_diverse", 0, "most action-dependent state"),
    ],
    "Enduro": [
        ("pre_reward", 6, "about to overtake a car"),
        ("pre_setback", 6, "about to collide"),
        ("argmax_diverse", 0, "most action-dependent state"),
    ],
}

DEFAULT_SELECTORS = [
    ("pre_reward", 6, "about to score"),
    ("pre_setback", 6, "about to lose a life"),
    ("argmax_diverse", 0, "most action-dependent state"),
]


def selectors_for(env_id, n_states):
    """Selector list for `env_id`, padded/truncated to `n_states` entries."""
    short = env_id.replace("NoFrameskip-v4", "").replace("-v5", "").replace("-v4", "")
    sels = list(SELECTOR_TABLE.get(short, DEFAULT_SELECTORS))
    while len(sels) < n_states:
        sels.append(("argmax_diverse", 0, "most action-dependent state"))
    return sels[:n_states]


def draw_rin_panel_scaled(
    frame, action_meanings, r_in_all, taken_action, scale=4, caption=None
):
    """visualize_r_in.draw_rin_panel at `scale`x resolution, with cell height fit to the action count."""
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h, w, _ = frame.shape
    n = len(action_meanings)

    pad = 5 * scale
    gap = 4 * scale
    title_h = 30 * scale

    cell_h = min(28 * scale, max(1, (h - title_h - pad) // n - gap))
    cell_h = max(cell_h, 10 * scale)
    panel_h = max(h, title_h + n * (cell_h + gap) + pad)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.32 * scale * min(1.0, cell_h / (28.0 * scale))
    labels = [f"> {name}: {val:.3f}" for name, val in zip(action_meanings, r_in_all)]
    text_w = max(cv2.getTextSize(t, font, font_scale, 1)[0][0] for t in labels)
    cell_w = text_w + 8 * scale
    panel_w = cell_w + 2 * pad

    panel = np.full((panel_h, panel_w, 3), 240, dtype=np.uint8)
    title_scale = min(
        0.5 * scale,
        0.5
        * scale
        * cell_w
        / max(1, cv2.getTextSize("r_in values", font, 0.5 * scale, 1)[0][0]),
    )
    cv2.putText(
        panel,
        "r_in values",
        (pad, int(20 * scale)),
        font,
        title_scale,
        (0, 0, 0),
        max(1, scale // 2),
        cv2.LINE_AA,
    )

    max_abs = max(float(np.abs(r_in_all).max()), 1e-6)
    for i in range(n):
        y = title_h + i * (cell_h + gap)
        norm_val = float(np.clip(r_in_all[i] / max_abs, -1, 1))
        # white->blue for positive, white->red for negative
        if norm_val >= 0:
            rgb = (
                int(255 * (1 - norm_val * 0.7)),
                int(255 * (1 - norm_val * 0.7)),
                255,
            )
        else:
            rgb = (
                255,
                int(255 * (1 + norm_val * 0.7)),
                int(255 * (1 + norm_val * 0.7)),
            )
        cv2.rectangle(panel, (pad, y), (pad + cell_w, y + cell_h), rgb[::-1], -1)

        taken = i == taken_action
        cv2.rectangle(
            panel,
            (pad, y),
            (pad + cell_w, y + cell_h),
            (0, 0, 0) if taken else (120, 120, 120),
            (2 if taken else 1) * max(1, scale // 2),
        )
        label = ("> " if taken else "  ") + f"{action_meanings[i]}: {r_in_all[i]:.3f}"
        cv2.putText(
            panel,
            label,
            (pad + 4 * scale, y + int(cell_h * 0.68)),
            font,
            font_scale,
            (255, 255, 255) if abs(norm_val) > 0.6 else (0, 0, 0),
            max(1, scale // 2),
            cv2.LINE_AA,
        )

    if panel_h > h:
        padded = np.full((panel_h, w, 3), 240, dtype=np.uint8)
        top = (panel_h - h) // 2
        padded[top : top + h] = frame
        frame = padded

    out = np.concatenate([frame, panel], axis=1)

    if caption:
        cap_scale = 0.34 * scale
        while (
            cv2.getTextSize(caption, font, cap_scale, 1)[0][0]
            > out.shape[1] - 8 * scale
            and cap_scale > 0.12
        ):
            cap_scale -= 0.02
        cap_h = 22 * scale
        strip = np.full((cap_h, out.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(
            strip,
            caption,
            (4 * scale, int(cap_h * 0.7)),
            font,
            cap_scale,
            (0, 0, 0),
            max(1, scale // 2),
            cv2.LINE_AA,
        )
        out = np.concatenate([out, strip], axis=0)
    return out


def contact_sheet(images, bg=255):
    """Stack panels side by side, top-aligned, padded to the tallest."""
    h = max(im.shape[0] for im in images)
    padded = []
    for im in images:
        if im.shape[0] < h:
            im = np.concatenate(
                [im, np.full((h - im.shape[0], im.shape[1], 3), bg, np.uint8)], axis=0
            )
        padded.append(im)
        padded.append(np.full((h, 8, 3), 200, np.uint8))
    return np.concatenate(padded[:-1], axis=1)


def load_models(run_dir, envs):
    """Load agent.pt and rin.pt, assuming the default shared encoder and per-action r_in head."""
    agent = AtariAgent(envs, shared_network=True).to(device).eval()
    r_in_fn = (
        IntrinsicRewardFn(envs, r_in_per_action=True, shared_rin_vex=True)
        .to(device)
        .eval()
    )
    # Aliases of shared modules that state_dict() leaves out.
    aliases = {"agent": ("network.",), "rin": ("value_model.",)}
    for path, module, what in (
        (os.path.join(run_dir, "agent.pt"), agent, "agent"),
        (os.path.join(run_dir, "rin.pt"), r_in_fn, "rin"),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no {what} weights at {path}")
        missing, unexpected = module.load_state_dict(
            torch.load(path, map_location=device), strict=False
        )
        missing = [k for k in missing if not k.startswith(aliases[what])]
        if missing:
            raise RuntimeError(
                f"{what} checkpoint {path} does not fit the architecture: "
                f"missing {sorted(missing)}"
            )
        if unexpected:
            print(f"  [info] {what}: {len(unexpected)} unused keys in checkpoint")
    return agent, r_in_fn


def rollout(
    env, agent, r_in_fns, total_steps, max_episodes, max_steps, deterministic, rng
):
    """Roll out `agent` for `total_steps`, recording each r_in in `r_in_fns`; returns per-step dicts."""
    steps = []
    for episode in range(max_episodes):
        if len(steps) >= total_steps:
            break
        obs, _ = env.reset()
        lives = env.unwrapped.ale.lives()
        done, t = False, 0
        ep_start = len(steps)
        while not done and t < max_steps:
            obs_t = torch.tensor(
                np.array(obs), dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                logits = agent.actor(agent.actor_network(obs_t / 255.0))
                if deterministic:
                    action = int(logits.argmax(dim=-1).item())
                else:
                    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                    action = int(rng.choice(len(probs), p=probs))
                r_ins = {
                    k: fn.get_intrinsic_reward(obs_t)[0].cpu().numpy()
                    for k, fn in r_in_fns.items()
                }

            frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            new_lives = env.unwrapped.ale.lives()

            steps.append(
                {
                    "frame": frame,
                    "r_in": r_ins,
                    "action": action,
                    "reward": float(reward),
                    "life_lost": new_lives < lives,
                    "episode": episode,
                }
            )
            lives = new_lives
            t += 1
        # EpisodicLifeEnv ends on a life loss before ale.lives() ticks.
        if done and len(steps) > ep_start and env.unwrapped.ale.lives() > 0:
            steps[-1]["life_lost"] = True
    return steps


def select_states(steps, key, selectors, min_gap):
    """Pick one step index per selector. Returns [(index, selector_name, label)]."""
    n = len(steps)
    r_in = np.stack([s["r_in"][key] for s in steps])
    spread = r_in.max(axis=1) - r_in.min(axis=1)
    argmax = r_in.argmax(axis=1)

    def within(pred, lookahead):
        """Indices in the same episode from which `pred` fires within `lookahead` steps."""
        out = set()
        for h in (i for i in range(n) if pred(steps[i])):
            for i in range(max(0, h - lookahead), h + 1):
                if steps[i]["episode"] == steps[h]["episode"]:
                    out.add(i)
        return out

    chosen = []

    def take(cands, name, label):
        cands = list(cands)
        for gap in (min_gap, max(1, min_gap // 2), 1):
            free = [i for i in cands if all(abs(i - c) >= gap for c, _, _ in chosen)]
            if free:
                chosen.append((int(max(free, key=lambda i: spread[i])), name, label))
                return True
        return False

    for name, lookahead, label in selectors:
        if name == "pre_reward":
            ok = take(within(lambda s: s["reward"] > 0, lookahead), name, label)
        elif name == "pre_setback":
            ok = take(
                within(lambda s: s["life_lost"] or s["reward"] < 0, lookahead),
                name,
                label,
            )
        elif name == "argmax_diverse":
            used = {argmax[i] for i, _, _ in chosen}
            ok = take([i for i in range(n) if argmax[i] not in used], name, label)
        else:
            raise ValueError(f"unknown selector {name}")
        if not ok:
            take(range(n), f"{name}:fallback", f"{label} (unavailable; top-contrast)")

    return chosen


def constancy_stats(steps, key):
    """How close to a constant function is this r_in over the trajectory?"""
    r_in = np.stack([s["r_in"][key] for s in steps])
    spread = r_in.max(axis=1) - r_in.min(axis=1)
    argmax = r_in.argmax(axis=1)
    modal, modal_n = Counter(argmax.tolist()).most_common(1)[0]
    patterns = Counter(map(tuple, np.sign(r_in).astype(int).tolist()))
    _, pattern_n = patterns.most_common(1)[0]
    return {
        "n_steps": len(steps),
        "mean_spread": float(spread.mean()),
        "median_spread": float(np.median(spread)),
        "mean_abs_r_in": float(np.abs(r_in).mean()),
        "mean_std_over_actions": float(r_in.std(axis=1).mean()),
        # 0 for a reward that is constant across states.
        "state_variation": float(r_in.std(axis=0).mean()),
        "argmax_constancy": float(modal_n / len(argmax)),
        "modal_argmax_action": int(modal),
        "sign_pattern_constancy": float(pattern_n / len(argmax)),
        "n_distinct_sign_patterns": len(patterns),
    }


def process(name, run_dir, steps, key, action_meanings, args, out_dir):
    """Write the per-state PNGs, contact sheet, manifest and stats for one run."""
    if len(steps) <= args.n_states:
        raise RuntimeError(
            f"[{name}] only {len(steps)} states collected; raise --total_steps "
            f"or --max_episodes"
        )
    os.makedirs(out_dir, exist_ok=True)
    selectors = selectors_for(args.env_id, args.n_states)
    chosen = select_states(steps, key, selectors, args.min_gap)
    stats = constancy_stats(steps, key)

    print(f"\n[{name}] {run_dir}")
    print(
        f"  state variation {stats['state_variation']:.3f} | "
        f"mean spread {stats['mean_spread']:.3f} | "
        f"mean |r_in| {stats['mean_abs_r_in']:.3f} | "
        f"argmax constancy {stats['argmax_constancy']:.2f} "
        f"({action_meanings[stats['modal_argmax_action']]}) | "
        f"sign-pattern constancy {stats['sign_pattern_constancy']:.2f} "
        f"over {stats['n_distinct_sign_patterns']} patterns"
    )

    manifest, sheet = [], []
    for k, (idx, sel, label) in enumerate(chosen, start=1):
        s = steps[idx]
        img = draw_rin_panel_scaled(
            s["frame"], action_meanings, s["r_in"][key], s["action"], scale=args.scale
        )
        path = os.path.join(out_dir, f"{name}_state{k}.png")
        cv2.imwrite(path, img)
        sheet.append(
            draw_rin_panel_scaled(
                s["frame"],
                action_meanings,
                s["r_in"][key],
                s["action"],
                scale=args.scale,
                caption=f"({chr(96 + k)}) {label}",
            )
        )
        manifest.append(
            {
                "file": os.path.basename(path),
                "step": idx,
                "selector": sel,
                "label": label,
                "action_taken": action_meanings[s["action"]],
                "r_in": {
                    a: round(float(v), 4)
                    for a, v in zip(action_meanings, s["r_in"][key])
                },
            }
        )
        print(f"  state{k}: step {idx:5d}  [{sel}]  {label}  -> {path}")

    cv2.imwrite(os.path.join(out_dir, f"{name}_states.png"), contact_sheet(sheet))
    with open(os.path.join(out_dir, f"{name}_manifest.json"), "w") as f:
        json.dump(
            {"run_dir": run_dir, "env_id": args.env_id, "states": manifest},
            f,
            indent=2,
        )
    with open(os.path.join(out_dir, f"{name}_stats.json"), "w") as f:
        json.dump({"run_dir": run_dir, **stats}, f, indent=2)
    return sheet


def survey(args):
    """Score several runs' r_in for how state-dependent it is."""
    dirs = list(args.survey_run_dirs)
    if args.lirpg_run_dir:
        dirs = [args.lirpg_run_dir] + dirs
    if args.env_id is None:
        args.env_id = os.path.basename(os.path.normpath(dirs[0])).split("__")[0]

    set_seed(args.seed)
    env = make_env(args.env_id)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    action_meanings = env.unwrapped.get_action_meanings()
    envs = type(
        "DummyVecEnv",
        (),
        {
            "single_action_space": env.action_space,
            "single_observation_space": env.observation_space,
        },
    )

    models = {}
    for d in dirs:
        try:
            models[os.path.basename(os.path.normpath(d))] = load_models(d, envs)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")

    rows = []
    if args.survey_policy == "shared":
        driver = os.path.basename(os.path.normpath(dirs[0]))
        steps = rollout(
            env,
            models[driver][0],
            {k: v[1] for k, v in models.items()},
            args.total_steps,
            args.max_episodes,
            args.max_steps,
            args.deterministic,
            np.random.default_rng(args.seed),
        )
        print(f"\n=== {args.env_id}: {len(steps)} shared states under {driver}")
        for key in models:
            rows.append({**constancy_stats(steps, key), "run": key})
    else:
        print(f"\n=== {args.env_id}: on-policy survey of {len(models)} runs")
        for key, (agent, r_in_fn) in models.items():
            env.reset(seed=args.seed)
            steps = rollout(
                env,
                agent,
                {key: r_in_fn},
                args.total_steps,
                args.max_episodes,
                args.max_steps,
                args.deterministic,
                np.random.default_rng(args.seed),
            )
            rows.append({**constancy_stats(steps, key), "run": key})
    env.close()

    rows.sort(key=lambda r: -r["state_variation"])

    print(
        f"{'state-var':>10} {'argmax-const':>13} {'sign-const':>11} "
        f"{'#patterns':>10} {'spread':>8}  run"
    )
    for r in rows:
        print(
            f"{r['state_variation']:10.3f} {r['argmax_constancy']:13.2f} "
            f"{r['sign_pattern_constancy']:11.2f} "
            f"{r['n_distinct_sign_patterns']:10d} {r['mean_spread']:8.3f}  {r['run']}"
        )

    out_dir = os.path.join(args.out_dir, args.env_id)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "survey.json")
    with open(path, "w") as f:
        json.dump(
            {
                "env_id": args.env_id,
                "survey_policy": args.survey_policy,
                "action_meanings": action_meanings,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"-> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lirpg_run_dir", type=str, default=None, help="intrinsic-only LIRPG run"
    )
    p.add_argument(
        "--dcpg_run_dir", type=str, default=None, help="intrinsic-only LIRPG+DCPG run"
    )
    p.add_argument("--out_dir", type=str, default="plots/r_in_states")
    p.add_argument(
        "--env_id", type=str, default=None, help="default: parsed from the run dir name"
    )
    p.add_argument("--n_states", type=int, default=3, help="states to render per run")
    p.add_argument(
        "--total_steps", type=int, default=3000, help="steps to collect per run"
    )
    p.add_argument(
        "--max_episodes", type=int, default=200, help="cap on episodes (lives)"
    )
    p.add_argument("--max_steps", type=int, default=2000, help="steps per episode")
    p.add_argument(
        "--min_gap", type=int, default=25, help="min steps between chosen states"
    )
    p.add_argument(
        "--scale", type=int, default=4, help="integer upscale of the rendered panel"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="argmax actions instead of sampling",
    )
    p.add_argument(
        "--rollout_policy",
        choices=["both", "dcpg", "lirpg"],
        default="both",
        help="'both': each run under its own policy; 'dcpg'/'lirpg': one rollout scored by both r_in",
    )
    p.add_argument(
        "--survey_run_dirs",
        type=str,
        nargs="+",
        default=None,
        help="score these runs' r_in for state-dependence and exit without rendering",
    )
    p.add_argument(
        "--survey_policy",
        choices=["own", "shared"],
        default="own",
        help="'own': each run under its own policy; 'shared': one rollout driven by the first run",
    )
    args = p.parse_args()

    if args.survey_run_dirs:
        return survey(args)

    runs = {
        k: v
        for k, v in (("lirpg", args.lirpg_run_dir), ("dcpg", args.dcpg_run_dir))
        if v
    }
    if not runs:
        p.error("give at least one of --lirpg_run_dir / --dcpg_run_dir")
    if args.rollout_policy != "both" and args.rollout_policy not in runs:
        p.error(
            f"--rollout_policy {args.rollout_policy} needs --{args.rollout_policy}_run_dir"
        )

    if args.env_id is None:
        args.env_id = os.path.basename(
            os.path.normpath(next(iter(runs.values())))
        ).split("__")[0]
    for run_dir in runs.values():
        got = os.path.basename(os.path.normpath(run_dir)).split("__")[0]
        assert got == args.env_id, f"{run_dir} is {got}, not {args.env_id}"

    out_dir = os.path.join(args.out_dir, args.env_id)
    print(f"=== {args.env_id} -> {out_dir}")

    set_seed(args.seed)
    env = make_env(args.env_id)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    action_meanings = env.unwrapped.get_action_meanings()
    print(f"  actions: {action_meanings}")

    envs = type(
        "DummyVecEnv",
        (),
        {
            "single_action_space": env.action_space,
            "single_observation_space": env.observation_space,
        },
    )
    models = {}
    for name, run_dir in runs.items():
        print(f"  loading {name}: {run_dir}")
        models[name] = load_models(run_dir, envs)

    sheets = {}
    if args.rollout_policy == "both":
        for name, run_dir in runs.items():
            agent, r_in_fn = models[name]
            env.reset(seed=args.seed)
            steps = rollout(
                env,
                agent,
                {name: r_in_fn},
                args.total_steps,
                args.max_episodes,
                args.max_steps,
                args.deterministic,
                np.random.default_rng(args.seed),
            )
            print(f"  [{name}] collected {len(steps)} steps")
            sheets[name] = process(
                name, run_dir, steps, name, action_meanings, args, out_dir
            )
    else:
        driver = args.rollout_policy
        steps = rollout(
            env,
            models[driver][0],
            {k: v[1] for k, v in models.items()},
            args.total_steps,
            args.max_episodes,
            args.max_steps,
            args.deterministic,
            np.random.default_rng(args.seed),
        )
        print(f"  collected {len(steps)} steps under the {driver} policy")
        for name, run_dir in runs.items():
            sheets[name] = process(
                name, run_dir, steps, name, action_meanings, args, out_dir
            )

    env.close()

    if len(sheets) > 1:
        combined = contact_sheet(
            [im for name in ("lirpg", "dcpg") if name in sheets for im in sheets[name]]
        )
        path = os.path.join(out_dir, f"{args.env_id}_figure.png")
        cv2.imwrite(path, combined)
        print(f"\ncombined preview -> {path}")
    print("Done!")


if __name__ == "__main__":
    main()
