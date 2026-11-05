#!/bin/bash
# Figures 7(a), 12 and 13: PPO, intrinsic-only PPO-LIRPG and PPO-LIRPG + DCPG on 49 Atari games.
source "$(dirname "$0")/common.sh" "$@"

D=runs-atari-complete
for g in "${ATARI49[@]}"; do for s in "${SEEDS[@]}"; do
    run $D "$(atari $g)" $s tuning_1_0_0 $(reward_flags 0)
    run $D "$(atari $g)" $s tuning_0_1_0 $(reward_flags 1)
    run $D "$(atari $g)" $s tuning_0_1_0-dcpg $(reward_flags 1) $DCPG_ATARI
    # The 7 analysis games use the random-policy runs of lambda_sweep.sh.
    [[ " ${ATARI7[*]} " == *" $g "* ]] || run $D "$(atari $g)" $s random_policy --random_policy
done; done
