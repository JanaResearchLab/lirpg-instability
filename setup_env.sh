#!/bin/bash
# Create one Python 3.10 venv per domain: .venv_atari, .venv_mujoco, .venv_procgen.
# Usage: bash setup_env.sh [atari|mujoco|procgen ...]   (default: all three)
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3.10}

setup() {
    local domain=$1 modules=$2
    (
        if [[ -n "${CC_CLUSTER:-}" ]]; then  # DRAC clusters
            [[ $domain == procgen ]] && module --force purge
            module load $modules
            virtualenv --no-download ".venv_$domain"
            grep -vE '^(numpy|scipy|opencv)' "requirements/$domain.txt" > ".venv_$domain/requirements.txt"
        else
            $PYTHON -m venv ".venv_$domain"
            cp "requirements/$domain.txt" ".venv_$domain/requirements.txt"
        fi
        source ".venv_$domain/bin/activate"
        pip install --upgrade pip
        pip install -r ".venv_$domain/requirements.txt"
        if [[ $domain == atari ]]; then
            pip install -r requirements/plotting.txt
            AutoROM --accept-license
        fi
    )
}

for domain in ${@:-atari mujoco procgen}; do
    case $domain in
        atari | mujoco) setup $domain "StdEnv/2023 python/3.10.13 opencv/4.10.0" ;;
        procgen) setup procgen "StdEnv/2020 gcc/9.3.0 python/3.10.2 opencv/4.8.0" ;;
        *) echo "unknown domain $domain" >&2; exit 1 ;;
    esac
done
