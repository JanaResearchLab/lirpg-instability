#!/bin/bash
# Paper figures from the runs of experiments/*.sh. Run from the repo root in the Atari venv.
# Usage: bash plotting/make_figures.sh [fig6 ...]   (default: every figure whose runs exist)
set -uo pipefail
cd "$(dirname "$0")/.."

P="python plotting"
RANDOM_ATARI='runs-atari-tuning/*__--random_policy--__*'
LOSS_TAGS="losses/inner_v_loss losses/inner_pg_loss losses/inner_joint_loss"
LIRPG_RIN='PPO-LIRPG ($r^\mathrm{in}$)'
PPO_REX='PPO ($r^\mathrm{ex}$)'

fig1_2() {
    $P/paper_figures.py fig1 fig2
}
fig3() {  # also writes <runs_dir>/best_runs.json for Tables 4-5
    $P/visualize_hyperparameters.py --runs_dir runs-atari-tuning --postfixes=""
    $P/visualize_hyperparameters.py --runs_dir runs-mujoco-tuning --postfixes=""
}
fig4() {
    $P/tb_events_to_report.py --runs_dir runs-atari-diagnostic --envs BeamRiderNoFrameskip-v4 \
        --settings "$LIRPG_RIN" --postfixes=",-freeze_0.3" --tags $LOSS_TAGS --log_scale
}
fig5() {
    $P/paper_figures.py fig5
}
fig6() {
    $P/paper_figures.py fig6 held_out   # held_out: Appendix B.3
}
fig8() {
    $P/link_runs.py runs-atari-tuning-dcpg "$RANDOM_ATARI"
    $P/visualize_hyperparameters.py --runs_dir runs-atari-tuning-dcpg --postfixes="-dcpg"
}
fig7() {
    $P/paper_figures.py fig7a fig7b fig7c fig15 fig16
    $P/plot_learning_curve_grid.py --runs_dir runs-atari-complete --n_cols 4 --games 0:28  # 12
    $P/plot_learning_curve_grid.py --runs_dir runs-atari-complete --n_cols 4 --games 28:   # 13
    $P/plot_learning_curve_grid.py --runs_dir runs-procgen --n_cols 5                     # 14
}
fig9() {
    $P/plot_alt_strategies_figure.py
}
fig10() {
    $P/paper_figures.py fig10
}
fig11() {
    $P/tb_events_to_report.py --runs_dir runs-atari-tuning --settings "$PPO_REX" "$LIRPG_RIN" \
        --postfixes="" --tags $LOSS_TAGS
}
fig17() {
    lirpg=$(ls -d runs-atari-tuning/PongNoFrameskip-v4__--tuning_0_1_0--__0__* | head -1)
    dcpg=$(ls -d runs-atari-dcpg-reg/PongNoFrameskip-v4__--tuning_0_1_0-ppg_vr_0.0_pr_1.0_tfreq_32--__0__* | head -1)
    $P/plot_r_in_states.py --lirpg_run_dir "$lirpg" --dcpg_run_dir "$dcpg" --out_dir plots/r_in_states
}
fig18() {
    $P/tb_events_to_report.py --runs_dir runs-atari-rnd --settings "PPO-RND" --tags charts/mean_r_rnd
    $P/tb_events_to_report.py --runs_dir runs-atari-rnd --settings "PPO-RND" --tags losses/inner_v_loss
}
legends() {
    $P/paper_figures.py legends
}
relu() {  # Appendix E.5
    $P/link_runs.py runs-mujoco-relu 'runs-mujoco-tuning/Inverted*__--random_policy--__*' \
        'runs-mujoco-tuning/Inverted*__--tuning_1_0_0--__*' 'runs-mujoco-tuning/Inverted*__--tuning_0_1_0--__*'
    $P/compare_methods.py --runs_dir runs-mujoco-relu
}

ALL=(fig1_2 fig3 fig4 fig5 fig6 fig7 fig8 fig9 fig10 fig11 fig17 fig18 legends relu)
for f in ${@:-${ALL[@]}}; do
    echo "===== $f"
    (set -e; $f)
    [[ $? -eq 0 ]] || echo "!!!!! $f failed (are its runs there?)"
done
