#!/bin/bash
# Table 11: DCPG value-phase frequency t_freq, with k_v = 8 t_freq.
source "$(dirname "$0")/common.sh" "$@"

for g in "${ATARI5[@]}"; do for s in "${SEEDS[@]}"; do
    for t in 1 8 16 64; do
        run runs-atari-dcpg-tfreq "$(atari $g)" $s "tuning_0_1_0-ppg_vr_0.0_pr_1.0_tfreq_$t" $(reward_flags 1) \
            $DCPG_ATARI --t_freq $t --k_v $((8 * t))
    done
done; done
