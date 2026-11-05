#!/bin/bash
# Figure 18: PPO with an RND bonus (r_ex + r_RND), using each game's tuned RND coefficient.
source "$(dirname "$0")/common.sh" "$@"

# game -> "<r_ex coef> <r_rnd coef>"
declare -A RND=([BeamRider]="0.99 0.01" [Breakout]="0.5 0.5" [Pong]="0.9 0.1")
for g in BeamRider Breakout Pong; do for s in "${SEEDS[@]}"; do
    c=(${RND[$g]})
    run runs-atari-rnd "$(atari $g)" $s "tuning_${c[0]}_0_${c[1]#0}" \
        --r_ex_coef ${c[0]} --r_in_coef 0 --r_rnd_coef ${c[1]} --no-meta_update
done; done
