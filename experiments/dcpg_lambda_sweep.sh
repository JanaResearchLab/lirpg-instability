#!/bin/bash
# Figure 8 and Table 4 (LIRPG + DCPG column): lambda_in sweep with DCPG on the 7 Atari games.
source "$(dirname "$0")/common.sh" "$@"

for g in "${ATARI7[@]}"; do for s in "${SEEDS[@]}"; do
    for l in 0 0.01 0.1 0.3 0.7 0.9 0.99 1; do
        run runs-atari-tuning-dcpg "$(atari $g)" $s "$(reward_tag $l)-dcpg" $(reward_flags $l) $DCPG_ATARI
    done
done; done
