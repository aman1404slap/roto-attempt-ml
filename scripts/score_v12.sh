#!/bin/bash
# One scoring pass per line of a job file, sequentially. Three of these run at once: a pass
# is ~2 min and CPU-bound in the rasteriser, so parallelism is nearly free on 16 cores, while
# the GPU side is one small inference per layer.
#
# Usage: scripts/score_v12.sh <jobfile>   (each line: <label> <report_v12.py args...>)
cd /home/aman/Projects/roto-v2 || exit 1
PY=~/.pyenv/shims/python3
while read -r label args; do
    [ -z "$label" ] && continue
    echo "SCORE start $label $(date +%H:%M:%S)"
    # shellcheck disable=SC2086
    $PY scripts/report_v12.py $args > "v1.2/logs/score_$label.log" 2>&1
    echo "SCORE done $label exit=$? $(date +%H:%M:%S) :: $(tail -3 v1.2/logs/score_$label.log | head -1)"
done < "$1"
echo "SCORE queue $1 finished"
