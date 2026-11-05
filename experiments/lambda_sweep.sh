#!/bin/bash
# Figures 1-3, 11, 17(a), Tables 4-5: lambda_in sweep on Atari and MuJoCo, plus random-policy runs.
source "$(dirname "$0")/common.sh" "$@"

LAMBDAS=(0 0.01 0.1 0.3 0.5 0.7 0.9 0.99 1)
for g in "${ATARI7[@]}"; do for s in "${SEEDS[@]}"; do
    for l in "${LAMBDAS[@]}"; do
        run runs-atari-tuning "$(atari $g)" $s "$(reward_tag $l)" $(reward_flags $l)
    done
    run runs-atari-tuning "$(atari $g)" $s random_policy --random_policy
done; done
for e in "${MUJOCO7[@]}"; do for s in "${SEEDS[@]}"; do
    for l in "${LAMBDAS[@]}"; do
        run runs-mujoco-tuning $e $s "$(reward_tag $l)" $(reward_flags $l)
    done
    run runs-mujoco-tuning $e $s random_policy --random_policy
done; done
