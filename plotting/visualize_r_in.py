"""Write videos of a run's learned r_in, V^in and Q^in alongside the game frames.

Usage: python plotting/visualize_r_in.py --run_dir <run> [--factors r_in_per_action v_in q_in_per_action]
"""

import argparse
import copy
import os
import sys

import cv2
import gymnasium as gym
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import AtariAgent
from intrinsic_reward_module import IntrinsicRewardFn

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FACTORS = ["r_in_per_action", "r_in_taken", "v_in", "q_in_per_action", "q_in_taken"]


def make_env(env_id):
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


def simulate_all_actions(env, n_actions, agent, r_in_all, gamma):
    """Deepcopy env for each action, step, and compute Q^in(s,a)."""
    q_values = []
    for a in range(n_actions):
        sim_env = copy.deepcopy(env)
        next_obs, _, terminated, truncated, _ = sim_env.step(a)
        done = terminated or truncated
        next_obs_t = torch.tensor(
            np.array(next_obs), dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            next_v = agent.get_value(next_obs_t).item()
        q_values.append(r_in_all[a] + gamma * next_v * (0.0 if done else 1.0))
        sim_env.close()
    return q_values


def simulate_single_action(env, action, agent, r_in_a, gamma):
    """Deepcopy env, step with one action, and compute Q^in(s,a)."""
    sim_env = copy.deepcopy(env)
    next_obs, _, terminated, truncated, _ = sim_env.step(action)
    done = terminated or truncated
    next_obs_t = torch.tensor(
        np.array(next_obs), dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        next_v = agent.get_value(next_obs_t).item()
    sim_env.close()
    return r_in_a + gamma * next_v * (0.0 if done else 1.0)


def _draw_bar(h, bar_width, fraction, color_pos, color_neg):
    """Vertical bar centred at mid-height. fraction in [-1, 1]."""
    bar = np.ones((h, bar_width, 3), dtype=np.uint8) * 255
    mid_y = h // 2
    bar_height = int(abs(fraction) * mid_y)
    if fraction > 0:
        bar[max(mid_y - bar_height, 0) : mid_y, :] = color_pos
    elif fraction < 0:
        bar[mid_y : min(mid_y + bar_height, h), :] = color_neg
    return bar


def _draw_label(bar, text, font_scale=0.35, thickness=1):
    """Put a small rotated label at the bottom of a bar column."""
    h, w, _ = bar.shape
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    txt_img = np.ones((th + 6, tw + 4, 3), dtype=np.uint8) * 255
    cv2.putText(
        txt_img,
        text,
        (2, th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    txt_img = cv2.rotate(txt_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rh, rw, _ = txt_img.shape
    x0 = max((w - rw) // 2, 0)
    y0 = h - rh - 2
    if y0 >= 0 and x0 + rw <= w and y0 + rh <= h:
        bar[y0 : y0 + rh, x0 : x0 + rw] = txt_img
    return bar


def draw_overlay(frame, bars_data):
    """Draw bars beside the frame; bars_data: [(label, value, full-height value, color_pos, color_neg)]."""
    h, w, _ = frame.shape
    bar_width = 20
    gap = 2

    parts = [frame]
    separator = np.ones((h, gap, 3), dtype=np.uint8) * 200

    for label, value, scale, color_pos, color_neg in bars_data:
        frac = value / scale if scale != 0 else 0.0
        frac = float(np.clip(frac, -1, 1))
        bar = _draw_bar(h, bar_width, frac, color_pos, color_neg)
        bar = _draw_label(bar, label)
        parts.extend([separator, bar])

    return np.concatenate(parts, axis=1)


def draw_rin_panel(frame, action_meanings, r_in_all, taken_action):
    """Draw a panel showing r_in values as color-coded cells with numbers."""
    h, w, _ = frame.shape
    cell_h = 28
    cell_w = 110
    padding = 5
    panel_w = cell_w + 2 * padding

    panel = np.ones((h, panel_w, 3), dtype=np.uint8) * 240

    cv2.putText(
        panel,
        "r_in values",
        (padding, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    max_abs = max(np.abs(r_in_all).max(), 1e-6)

    y_start = 30
    for i in range(len(action_meanings)):
        y = y_start + i * (cell_h + 4)
        if y + cell_h > h - 5:
            break

        val = r_in_all[i]
        norm_val = np.clip(val / max_abs, -1, 1)

        # white->blue for positive, white->red for negative
        if norm_val >= 0:
            r_c = int(255 * (1 - norm_val * 0.7))
            g_c = int(255 * (1 - norm_val * 0.7))
            b_c = 255
        else:
            r_c = 255
            g_c = int(255 * (1 + norm_val * 0.7))
            b_c = int(255 * (1 + norm_val * 0.7))

        color = (b_c, g_c, r_c)  # BGR

        cv2.rectangle(panel, (padding, y), (padding + cell_w, y + cell_h), color, -1)

        border_thickness = 2 if i == taken_action else 1
        border_color = (0, 0, 0) if i == taken_action else (120, 120, 120)
        cv2.rectangle(
            panel,
            (padding, y),
            (padding + cell_w, y + cell_h),
            border_color,
            border_thickness,
        )

        name = action_meanings[i]
        prefix = "> " if i == taken_action else "  "
        label = f"{prefix}{name}: {val:.3f}"
        text_color = (255, 255, 255) if abs(norm_val) > 0.6 else (0, 0, 0)
        cv2.putText(
            panel,
            label,
            (padding + 4, y + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return np.concatenate([frame, panel], axis=1)


def build_bars(data, action_meanings, action_indices, factors):
    """Build bar chart entries for the given action indices and factors."""
    CLR_RIN_POS, CLR_RIN_NEG = (255, 0, 0), (0, 0, 255)
    CLR_V_POS, CLR_V_NEG = (0, 180, 0), (0, 0, 200)
    CLR_Q_POS, CLR_Q_NEG = (200, 120, 0), (0, 0, 200)
    MAX_VALUE = 10.0

    bars = []
    a = data["action"]
    r_in_all = data["r_in_all"]
    v_in = data["v_in"]
    q_in_all = data["q_in_all"]
    q_in_taken = data["q_in_taken"]

    if "r_in_per_action" in factors:
        for i in action_indices:
            name = action_meanings[i]
            label = f"*{name}" if i == a else name
            bars.append((label, r_in_all[i], 1.0, CLR_RIN_POS, CLR_RIN_NEG))

    if "r_in_taken" in factors:
        bars.append(("r_in", r_in_all[a], 1.0, CLR_RIN_POS, CLR_RIN_NEG))

    if "v_in" in factors:
        bars.append(("V_in", v_in, MAX_VALUE, CLR_V_POS, CLR_V_NEG))

    if "q_in_per_action" in factors:
        for i in action_indices:
            name = action_meanings[i]
            label = f"Q:*{name}" if i == a else f"Q:{name}"
            if q_in_all is not None:
                bars.append((label, q_in_all[i], MAX_VALUE, CLR_Q_POS, CLR_Q_NEG))

    if "q_in_taken" in factors:
        if q_in_taken is not None:
            bars.append(("Q_in", q_in_taken, MAX_VALUE, CLR_Q_POS, CLR_Q_NEG))

    return bars


def write_video(frames, path, framerate):
    """Write a list of BGR frames to an mp4 file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(path, fourcc, framerate, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"  Video saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize intrinsic reward factors for Atari agents"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Directory containing agent.pt and rin.pt",
    )
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--framerate", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--factors",
        type=str,
        nargs="+",
        choices=FACTORS,
        default=["q_in_per_action"],
        help="Which factors to visualize in bar-chart videos",
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Max steps per episode"
    )
    args = parser.parse_args()
    args.env_id = os.path.basename(args.run_dir).split("__")[0]
    agent_path = os.path.join(args.run_dir, "agent.pt")
    rin_path = os.path.join(args.run_dir, "rin.pt")

    if not os.path.isfile(agent_path):
        print(f"Warning: agent weights not found at {agent_path}")
    if not os.path.isfile(rin_path):
        print(f"Warning: rin weights not found at {rin_path}")

    env = make_env(args.env_id)
    n_actions = env.action_space.n
    action_meanings = env.unwrapped.get_action_meanings()
    print(args.env_id, "action meanings:", action_meanings)

    fire_actions = {i for i, name in enumerate(action_meanings) if "FIRE" in name}
    non_fire_actions = [i for i in range(n_actions) if i not in fire_actions]
    all_actions = list(range(n_actions))
    print(f"  Fire actions: {[action_meanings[i] for i in fire_actions]}")
    print(f"  Non-fire actions: {[action_meanings[i] for i in non_fire_actions]}")

    envs = type(
        "DummyVecEnv",
        (),
        {
            "single_action_space": env.action_space,
            "single_observation_space": env.observation_space,
        },
    )
    agent = AtariAgent(envs).to(device)
    r_in_fn = IntrinsicRewardFn(envs, r_in_per_action=True, shared_rin_vex=True).to(
        device
    )

    # Aliases of shared modules that state_dict() leaves out.
    for path, module, alias in (
        (agent_path, agent, "network."),
        (rin_path, r_in_fn, "value_model."),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no weights at {path}")
        missing, _ = module.load_state_dict(
            torch.load(path, map_location=device), strict=False
        )
        missing = [k for k in missing if not k.startswith(alias)]
        if missing:
            raise RuntimeError(
                f"{path} does not fit the architecture: missing {sorted(missing)}"
            )

    need_q_all = "q_in_per_action" in args.factors
    need_q_taken = "q_in_taken" in args.factors

    raw_frames = []
    step_data = []

    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        obs_t = torch.tensor(
            np.array(obs), dtype=torch.float32, device=device
        ).unsqueeze(0)
        done = False
        step_count = 0
        while not done and step_count < args.max_steps:
            with torch.no_grad():
                action, _, _, value, _ = agent.get_action_and_value(obs_t)
                r_in_out = r_in_fn.get_intrinsic_reward(obs_t)  # (1, n_actions)
                r_in_all = r_in_out[0].cpu().numpy()
                v_in = value.item()

            a = action.item()

            q_in_all = None
            q_in_taken_val = None
            if need_q_all:
                q_in_all = simulate_all_actions(
                    env, n_actions, agent, r_in_all, args.gamma
                )
                q_in_taken_val = q_in_all[a]
            elif need_q_taken:
                q_in_taken_val = simulate_single_action(
                    env, a, agent, r_in_all[a], args.gamma
                )

            next_obs, _, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            raw_frames.append(frame)

            step_data.append(
                {
                    "r_in_all": r_in_all,
                    "action": a,
                    "v_in": v_in,
                    "q_in_all": q_in_all,
                    "q_in_taken": q_in_taken_val,
                }
            )

            obs_t = torch.tensor(
                np.array(next_obs), dtype=torch.float32, device=device
            ).unsqueeze(0)
            step_count += 1

    env.close()

    out_dir = "videos"
    os.makedirs(out_dir, exist_ok=True)
    run_name = os.path.basename(os.path.abspath(args.run_dir))

    print("Generating all-actions video...")
    frames_all = []
    for frame, data in zip(raw_frames, step_data):
        bars = build_bars(data, action_meanings, all_actions, args.factors)
        frames_all.append(draw_overlay(frame, bars))
    write_video(
        frames_all, os.path.join(out_dir, f"{run_name}_all_actions.mp4"), args.framerate
    )

    print("Generating no-fire video...")
    frames_no_fire = []
    for frame, data in zip(raw_frames, step_data):
        bars = build_bars(data, action_meanings, non_fire_actions, args.factors)
        frames_no_fire.append(draw_overlay(frame, bars))
    write_video(
        frames_no_fire, os.path.join(out_dir, f"{run_name}_no_fire.mp4"), args.framerate
    )

    print("Generating r_in values video...")
    frames_rin = []
    for frame, data in zip(raw_frames, step_data):
        frames_rin.append(
            draw_rin_panel(frame, action_meanings, data["r_in_all"], data["action"])
        )
    write_video(
        frames_rin, os.path.join(out_dir, f"{run_name}_rin_values.mp4"), args.framerate
    )

    print("Done!")


if __name__ == "__main__":
    main()
