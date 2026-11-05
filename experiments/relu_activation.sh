#!/bin/bash
# Appendix E.5: ReLU instead of tanh for the r_in output on the inverted pendulum tasks.
source "$(dirname "$0")/common.sh" "$@"

for e in InvertedPendulum-v4 InvertedDoublePendulum-v4; do for s in "${SEEDS[@]}"; do
    run runs-mujoco-relu $e $s tuning_0_1_0-relu $(reward_flags 1) --r_in_activation relu
done; done
