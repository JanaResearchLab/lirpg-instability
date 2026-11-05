#!/bin/bash
# Figures 10 and 17(b-d): DCPG regularization ablation over (beta_v, beta_pi) on the 7 Atari games.
source "$(dirname "$0")/common.sh" "$@"

D=runs-atari-dcpg-reg
for g in "${ATARI7[@]}"; do for s in "${SEEDS[@]}"; do
    for bv in 0.0 1.0; do for bp in 0.0 1.0; do
        run $D "$(atari $g)" $s "tuning_0_1_0-ppg_vr_${bv}_pr_${bp}_tfreq_32" $(reward_flags 1) \
            $DCPG_ATARI --beta_v $bv --beta_pi $bp
    done; done
done; done
