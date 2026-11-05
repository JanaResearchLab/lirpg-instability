#!/bin/bash
# Figures 7(c), 15 and 16: practice-match with and without DCPG, settings from Table 7.
source "$(dirname "$0")/common.sh" "$@"

declare -A STEPS=([BeamRider]=960 [Breakout]=180 [Enduro]=2600 [Pong]=600)
declare -A BUFFER=([BeamRider]=19200 [Breakout]=1800 [Enduro]=26000 [Pong]=9000)
D=runs-atari-practice
for g in BeamRider Breakout Enduro Pong; do for s in "${SEEDS[@]}"; do
    P="--r_ex_coef 1 --r_in_coef 1 --meta_update --practice_steps ${STEPS[$g]}"
    P="$P --practice_buffer_size ${BUFFER[$g]} --practice_use_agent_vex --max_match_timesteps 10000000"
    run $D "$(atari $g)" $s practice $P
    run $D "$(atari $g)" $s practice_dcpg $P $DCPG_ATARI
done; done
