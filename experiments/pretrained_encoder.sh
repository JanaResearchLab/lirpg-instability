#!/bin/bash
# Table 10: intrinsic-only runs with the encoder of the best extrinsic-only run. Needs lambda_sweep.sh.
source "$(dirname "$0")/common.sh" "$@"

D=runs-atari-comparison-pretrained_init
for g in "${ATARI7[@]}"; do
    enc=$(python3 "$(dirname "$0")/best_extrinsic_encoder.py" runs-atari-tuning "$(atari $g)")
    for s in "${SEEDS[@]}"; do
        run $D "$(atari $g)" $s pretrained_policy_only $(reward_flags 1) \
            --shared_networks rin-vex --agent_base_model_path $enc --agent_base_model_target policy
    done
done
