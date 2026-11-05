#!/bin/bash
# Figure 5: shared vs separate encoders on Atari.
source "$(dirname "$0")/common.sh" "$@"

D=runs-atari-comparison-unshared
for g in "${ATARI7[@]}"; do for s in "${SEEDS[@]}"; do
    lm=${ATARI_MIXED_LAMBDA[$g]}
    for l in 0 $lm 1; do
        run $D "$(atari $g)" $s "$(reward_tag $l)-shared" $(reward_flags $l)
    done
    for l in 0 0.01 1; do
        run $D "$(atari $g)" $s "$(reward_tag $l)-unshared" $(reward_flags $l) --shared_networks=
    done
done; done
