#!/bin/bash
# Everything after the training queue drains, in the order the report needs it.
#
# One script rather than a list in a README, because the order matters in three places and
# getting it wrong produces a table that is quietly stale rather than an error:
#   - the operating points must be swept BEFORE the gate rows are scored, or the gate is read
#     at a tolerance chosen for a different model;
#   - every run must be re-scored under the CURRENT columns, because rows written earlier in a
#     session can predate a metric (summarise_v13.py prints those as `_n/m_`, but the final
#     tables want them present);
#   - gates.py and summarise_v13.py read what the scoring wrote, so they come last.
#
# Usage: scripts/finalise_v13.sh [candidate]        (default: v002_final)
set -u
cd /home/aman/Projects/roto-v2 || exit 1
PY=~/.pyenv/shims/python3
CAND="${1:-v002_final}"
step () { echo; echo "=============== $* ($(date +%H:%M:%S)) ==============="; }

step "1. operating points, per model and per motion source"
for run in v002_final v002_keytime; do
    [ -f "v1.3/runs/$run/model.pt" ] || continue
    PYTHONPATH=src $PY -u scripts/sweep_v13.py --run "$run" --out v1.3/results \
        > "v1.3/logs/sweep_$run.log" 2>&1 &
    PYTHONPATH=src $PY -u scripts/sweep_v13.py --run "$run" --motion predicted --out v1.3/results \
        > "v1.3/logs/sweep_${run}_e2e.log" 2>&1 &
done
wait
grep -h "best IoU with keys" v1.3/logs/sweep_v002_*.log 2>/dev/null

step "2. score every row under the current columns"
scripts/score_v13.sh v1.3/logs/score_jobs.txt

step "3. the key-timing baselines, per run with a head"
for run in v002_keytime v002_keytime_s1 v002_keytime_probe v002_keytime_probe2; do
    [ -f "v1.3/runs/$run/model.pt" ] || continue
    PYTHONPATH=src $PY scripts/exp_key_baselines.py --run "$run" \
        > "v1.3/logs/key_baselines_$run.log" 2>&1
    tail -5 "v1.3/logs/key_baselines_$run.log"
done

step "4. the noise floor and the re-baseline"
PYTHONPATH=src $PY scripts/exp_noise_floor.py 2>&1 | tail -20
PYTHONPATH=src $PY scripts/rebaseline_v13.py 2>&1 | tail -20

step "5. the ledger -- must be green before anything is quoted"
PYTHONPATH=src $PY scripts/ledger.py 2>&1 | tail -22

step "6. the gates, on the candidate"
PYTHONPATH=src $PY scripts/gates.py --candidate "$CAND" \
    --seeds "${CAND}_gate" "${CAND}_s1_gate" --suffix _e2e_refit 2>&1 | tail -30

step "7. figures and the track-end diagnostic"
PYTHONPATH=src $PY scripts/report_v13.py --run "$CAND" --smooth 1 --tol 1.0 --refit \
    --label "${CAND}_ship" --figures --out v1.3/results > v1.3/logs/score_figures.log 2>&1
PYTHONPATH=src $PY scripts/fig_worst_frames.py --run "${CAND}_ship" \
    --score "v1.3/results/score_${CAND}_ship_refit.json" 2>&1 | tail -4
PYTHONPATH=src $PY scripts/exp_track_end.py --run "$CAND" 2>&1 | tail -6

step "8. the deliverable: real .sfx files at the shipping operating point"
PYTHONPATH=src $PY scripts/write_sfx_v13.py --run "$CAND" --smooth 1 --tol 1.0 --refit 2>&1 | tail -4

step "9. the tables"
PYTHONPATH=src $PY scripts/summarise_v13.py > /dev/null 2>&1 \
    && echo "-> v1.3/results/results.md"
echo; echo "finalise done $(date +%H:%M:%S)"
