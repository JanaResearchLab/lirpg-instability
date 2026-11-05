#!/bin/bash
# Tables 8 and 13, Appendix D: encoder gradient statistics logged during training.
source "$(dirname "$0")/common.sh" "$@"

for g in "${ATARI7[@]}"; do for s in "${SEEDS[@]}"; do
    D=runs-atari-diagnostic-grad_stats_compare
    run $D "$(atari $g)" $s tuning_0_1_0-grad_stats $(reward_flags 1) --log_grad_stats
    run $D "$(atari $g)" $s tuning_1_0_0-grad_stats_ppo $(reward_flags 0) --log_grad_stats
done; done
for g in "${ATARI_EXCEPTIONS[@]}"; do for s in "${SEEDS[@]}"; do
    run runs-atari-diagnostic-grad_stats "$(atari $g)" $s tuning_0_1_0-grad_stats $(reward_flags 1) \
        --log_grad_stats
done; done
