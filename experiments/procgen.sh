#!/bin/bash
# Figures 7(b) and 14: PPO, intrinsic-only PPO-LIRPG and PPO-LIRPG + DCPG on 5 ProcGen games.
source "$(dirname "$0")/common.sh" "$@"

D=runs-procgen
for g in "${PROCGEN5[@]}"; do for s in "${SEEDS[@]}"; do
    run $D "$(procgen $g)" $s tuning_1_0_0 $(reward_flags 0)
    run $D "$(procgen $g)" $s tuning_0_1_0 $(reward_flags 1)
    run $D "$(procgen $g)" $s tuning_0_1_0-dcpg $(reward_flags 1) $DCPG_PROCGEN
done; done
