#!/bin/bash
# One scoring pass per line of a job file, sequentially. Several of these can run at once: a
# pass is CPU-bound in the rasteriser (one small GPU inference per layer, then 1,810 renders),
# so parallelism is nearly free on 16 cores -- and it overlaps with training on the card.
#
# Usage: scripts/score_v13.sh <jobfile>   (each line: <label> <report_v13.py args...>)
cd /home/aman/Projects/roto-v2 || exit 1
PY=~/.pyenv/shims/python3
mkdir -p v1.3/logs
while read -r label args; do
    [ -z "$label" ] && continue
    case "$label" in \#*) continue ;; esac
    echo "SCORE start $label $(date +%H:%M:%S)"
    # shellcheck disable=SC2086
    PYTHONPATH=src $PY scripts/report_v13.py $args > "v1.3/logs/score_$label.log" 2>&1
    echo "SCORE done $label exit=$? $(date +%H:%M:%S) :: $(grep -m1 'soft IoU ' "v1.3/logs/score_$label.log" | tail -c 120)"
done < "$1"
echo "SCORE queue $1 finished"
