#!/bin/bash
# Figure 4: loss diagnostics on BeamRider (one seed), with and without freezing r_in after 30% of training.
source "$(dirname "$0")/common.sh" "$@"

s=0
run runs-atari-diagnostic "$(atari BeamRider)" $s tuning_0_1_0 $(reward_flags 1) --log_grad_stats
run runs-atari-diagnostic "$(atari BeamRider)" $s tuning_0_1_0-freeze_0.3 $(reward_flags 1) \
    --log_grad_stats --freeze_rin_frac 0.3
