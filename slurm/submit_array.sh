#!/bin/bash
# Submit the commands printed by an experiments/*.sh script as one SLURM job array per domain.
# Usage: bash slurm/submit_array.sh commands.txt [extra sbatch arguments]
# Env overrides: SBATCH_ACCOUNT, MODULES_/VENV_/TIME_/CPUS_/MEM_/GPUS_<DOMAIN>, MAX_PARALLEL, DRY_RUN=1.
set -euo pipefail

cmds=$(readlink -f "$1")
shift
repo=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$repo/slurm_logs"

MODULES_atari=${MODULES_ATARI:-"StdEnv/2023 python/3.10.13 opencv/4.10.0"}
MODULES_mujoco=${MODULES_MUJOCO:-"StdEnv/2023 python/3.10.13 opencv/4.10.0"}
MODULES_procgen=${MODULES_PROCGEN:-"StdEnv/2020 gcc/9.3.0 python/3.10.2 opencv/4.8.0"}
VENV_atari=${VENV_ATARI:-"$repo/.venv_atari"}
VENV_mujoco=${VENV_MUJOCO:-"$repo/.venv_mujoco"}
VENV_procgen=${VENV_PROCGEN:-"$repo/.venv_procgen"}
TIME_atari=${TIME_ATARI:-"24:00:00"}; CPUS_atari=${CPUS_ATARI:-8}; MEM_atari=${MEM_ATARI:-32G}; GPUS_atari=${GPUS_ATARI:-1}
TIME_procgen=${TIME_PROCGEN:-"16:00:00"}; CPUS_procgen=${CPUS_PROCGEN:-12}; MEM_procgen=${MEM_PROCGEN:-64G}; GPUS_procgen=${GPUS_PROCGEN:-1}
TIME_mujoco=${TIME_MUJOCO:-"8:00:00"}; CPUS_mujoco=${CPUS_MUJOCO:-2}; MEM_mujoco=${MEM_MUJOCO:-4G}; GPUS_mujoco=${GPUS_MUJOCO:-0}

tmp=$(mktemp -d)
grep -E '^python ppo_rin.py .*--env_id [A-Za-z0-9]+NoFrameskip-v4' "$cmds" > "$tmp/atari" || true
grep -E '^python ppo_rin.py .*--env_id procgen-' "$cmds" > "$tmp/procgen" || true
grep -E '^python ppo_rin.py' "$cmds" | grep -v -E -- '--env_id ([A-Za-z0-9]+NoFrameskip-v4|procgen-)' > "$tmp/mujoco" || true

for domain in atari procgen mujoco; do
    list="$tmp/$domain"
    n=$(wc -l < "$list")
    [[ $n -eq 0 ]] && continue
    keep="$repo/slurm_logs/commands_${domain}_$(date +%Y%m%d_%H%M%S).txt"
    cp "$list" "$keep"
    modules_var=MODULES_$domain; venv_var=VENV_$domain
    time_var=TIME_$domain; cpus_var=CPUS_$domain; mem_var=MEM_$domain; gpus_var=GPUS_$domain
    gpu_flag=()
    [[ ${!gpus_var} -gt 0 ]] && gpu_flag=(--gpus-per-node="${!gpus_var}")
    # ProcGen's module stack must be loaded from a clean environment.
    purge=""
    [[ $domain == procgen ]] && purge="module --force purge"
    submit=(sbatch)
    [[ ${DRY_RUN:-0} == 1 ]] && submit=(echo sbatch)
    "${submit[@]}" --job-name="lirpg-$domain" --array="1-$n%${MAX_PARALLEL:-50}" \
        --time="${!time_var}" --cpus-per-task="${!cpus_var}" --mem="${!mem_var}" ${gpu_flag[@]+"${gpu_flag[@]}"} \
        --output="$repo/slurm_logs/%x_%A_%a.out" "$@" <<EOF
#!/bin/bash
set -e
$purge
module load ${!modules_var}
source ${!venv_var}/bin/activate
cd "$repo"
cmd=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "$keep")
echo "\$cmd"
eval "\$cmd"
EOF
    echo "submitted $n $domain run(s); commands in $keep"
done
