#!/bin/bash
# Table 9: per-phase policy (KL) and value (0.5 * MSE) changes under DCPG, intrinsic-only.
source "$(dirname "$0")/common.sh" "$@"

for g in "${ATARI7[@]}"; do for s in "${SEEDS[@]}"; do
    run runs-atari-dcpg-phase-drift "$(atari $g)" $s tuning_0_1_0-dcpg_phasic $(reward_flags 1) \
        $DCPG_ATARI --log_grad_stats
done; done
