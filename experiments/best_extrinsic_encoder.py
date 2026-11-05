"""Print the actor_network.pt of the extrinsic-only seed with the best final 10% return.

Usage: python experiments/best_extrinsic_encoder.py <runs_dir> <env_id>
"""

import glob
import os
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def final_return(run_dir):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    returns = [e.value for e in ea.Scalars("charts/episodic_return")]
    tail = returns[-max(1, len(returns) // 10):]
    return float(np.mean(tail))


def main():
    runs_dir, env_id = sys.argv[1], sys.argv[2]
    runs = sorted(glob.glob(os.path.join(runs_dir, f"{env_id}__--tuning_1_0_0--__*")))
    runs = [r for r in runs if os.path.exists(os.path.join(r, "actor_network.pt"))]
    if not runs:
        sys.exit(f"no extrinsic-only run with actor_network.pt for {env_id} in {runs_dir}")
    best = max(runs, key=final_return)
    print(os.path.join(best, "actor_network.pt"))


if __name__ == "__main__":
    main()
