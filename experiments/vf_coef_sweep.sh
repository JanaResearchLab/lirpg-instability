#!/bin/bash
# Table 12: value-loss coefficient lambda_V for intrinsic-only PPO-LIRPG.
source "$(dirname "$0")/common.sh" "$@"

for g in "${ATARI5[@]}"; do for s in "${SEEDS[@]}"; do
    for vf in 0.01 0.05 0.1 0.25 0.5; do
        run runs-atari-vf_coef_sweep "$(atari $g)" $s "tuning_0_1_0-vf_$vf" $(reward_flags 1) --vf_coef $vf
    done
done; done
