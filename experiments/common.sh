# Sourced by experiments/*.sh. By default each `run` prints its command for any launcher.
# With --run the commands are executed one after another instead.

set -euo pipefail

MODE=print
[[ "${1:-}" == "--run" ]] && MODE=run

SEEDS=(0 1 2)

ATARI7=(BeamRider Breakout Enduro Pong Qbert Seaquest SpaceInvaders)
ATARI5=(BeamRider Breakout Pong Seaquest SpaceInvaders)
# Table 13: games where intrinsic-only LIRPG is already competitive.
ATARI_EXCEPTIONS=(DemonAttack KungFuMaster Centipede Asteroids Pitfall IceHockey)
ATARI49=(Alien Amidar Assault Asterix Asteroids Atlantis BankHeist BattleZone BeamRider
         Bowling Boxing Breakout Centipede ChopperCommand CrazyClimber DemonAttack DoubleDunk
         Enduro FishingDerby Freeway Frostbite Gopher Gravitar IceHockey Jamesbond Kangaroo
         Krull KungFuMaster MontezumaRevenge MsPacman NameThisGame Pitfall Pong PrivateEye
         Qbert Riverraid RoadRunner Robotank Seaquest SpaceInvaders StarGunner Tennis TimePilot
         Tutankham UpNDown Venture VideoPinball WizardOfWor Zaxxon)
MUJOCO7=(HalfCheetah-v4 Hopper-v4 Walker2d-v4 InvertedPendulum-v4 InvertedDoublePendulum-v4
         Reacher-v4 Swimmer-v4)
PROCGEN5=(coinrun starpilot caveflyer dodgeball fruitbot)

# Tuned lambda_in per game for the mixed-reward setting (Table 4).
declare -A ATARI_MIXED_LAMBDA=([BeamRider]=0.3 [Breakout]=0.01 [Enduro]=0.01 [Pong]=0.1
                               [Qbert]=0.01 [Seaquest]=0.1 [SpaceInvaders]=0.01)

# DCPG (Table 6).
DCPG_ATARI="--use_ppg --t_freq 32 --t_off 16 --k_v 256 --m_v 256 --beta_pi 1 --beta_v 0"
DCPG_PROCGEN="--use_ppg --t_freq 32 --t_off 16 --k_v 64 --m_v 256 --beta_pi 1 --beta_v 0"

atari() { echo "${1}NoFrameskip-v4"; }
procgen() { echo "procgen-${1}"; }

# lambda_in -> "<exp_name tag> <lambda_ex>" (lambda_ex = 1 - lambda_in).
declare -A _LAMBDA=([0]="tuning_1_0_0 1" [0.01]="tuning_0.99_.01_0 0.99" [0.1]="tuning_0.9_.1_0 0.9"
                    [0.3]="tuning_0.7_.3_0 0.7" [0.5]="tuning_0.5_.5_0 0.5" [0.7]="tuning_0.3_.7_0 0.3"
                    [0.9]="tuning_0.1_.9_0 0.1" [0.99]="tuning_0.01_.99_0 0.01" [1]="tuning_0_1_0 0")

reward_tag() { local v=(${_LAMBDA[$1]}); echo "${v[0]}"; }

reward_flags() {
    local v=(${_LAMBDA[$1]})
    if [[ $1 == 0 ]]; then
        echo "--r_ex_coef 1 --r_in_coef 0 --no-meta_update"
    else
        echo "--r_ex_coef ${v[1]} --r_in_coef $1 --meta_update"
    fi
}

# run <runs_dir> <env_id> <seed> <exp_name> [flags...]
run() {
    local runs_dir=$1 env_id=$2 seed=$3 exp_name=$4
    shift 4
    local cmd="python ppo_rin.py --env_id $env_id --seed $seed --runs_dir $runs_dir --exp_name=--${exp_name}-- $*"
    if [[ $MODE == run ]]; then
        eval "$cmd"
    else
        echo "$cmd"
    fi
}
