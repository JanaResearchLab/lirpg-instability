#!/bin/bash
# Figure 9: alternate strategies for intrinsic-only PPO-LIRPG on the 7 Atari games.
source "$(dirname "$0")/common.sh" "$@"

D=runs-atari-alternate-strategies
I=$(reward_flags 1)
for g in "${ATARI7[@]}"; do for s in "${SEEDS[@]}"; do
    e=$(atari $g)
    run $D $e $s tuning_0_1_0-r_in_if_r_ex $I --handle_sparse_rex r_in_if_r_ex
    run $D $e $s tuning_0_1_0-scale_rin_dens $I --handle_sparse_rex scale_r_in_by_density
    run $D $e $s tuning_0_1_0-vcoef_0.05 $I --vf_coef 0.05
    run $D $e $s tuning_0_1_0-gamma_0.9 $I --gamma 0.9 --gamma_outer 0.99
    run $D $e $s tuning_0_1_0-popart $I --use_popart
    run $D $e $s tuning_0_1_0-target_rin_tau0.02 $I --use_target_rin --target_rin_tau 0.02
    run $D $e $s tuning_0_1_0-pcgrad $I --pcgrad
    run $D $e $s tuning_0_1_0-gradnorm $I --gradnorm
done; done
