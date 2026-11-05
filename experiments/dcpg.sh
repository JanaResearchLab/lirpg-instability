#!/bin/bash
# Appendix B.3: the mixed-reward comparison of Figure 6 on held-out seeds 10-12.
source "$(dirname "$0")/common.sh" "$@"

D=runs-atari-heldout-mixed
for g in "${ATARI7[@]}"; do for s in 10 11 12; do
    lm=${ATARI_MIXED_LAMBDA[$g]}
    run $D "$(atari $g)" $s "$(reward_tag $lm)" $(reward_flags $lm)
    run $D "$(atari $g)" $s "$(reward_tag 0.5)-dcpg" $(reward_flags 0.5) $DCPG_ATARI
done; done
