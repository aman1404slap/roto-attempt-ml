#!/bin/bash
# Score each queued run as soon as it finishes, instead of waiting for the queue to drain.
#
# Scoring is CPU-bound in the rasteriser -- one small GPU inference per layer, then 1,810
# renders -- so it overlaps with the next run's training on the card almost for free. Wall
# clock is not a variable this round claims (v1.2 S7 makes the same point), and the
# alternative is 4 hours of an idle CPU followed by half an hour of an idle GPU.
#
# Two passes per run, and they are different questions:
#   base           the run at v1's own operating point -- the row the ladder table quotes
#   the gate       the constrained operating point with predicted motion end to end, which
#                  is what the plan's S4 gates: a delivery has neither the artist's motion
#                  track nor permission to place keys wherever rendered IoU likes them
#
# Usage: scripts/autoscore_v13.sh [seconds-between-polls]
cd /home/aman/Projects/roto-v2 || exit 1
PY=~/.pyenv/shims/python3
POLL="${1:-60}"
mkdir -p v1.3/logs v1.3/results

score () {   # <label> <args...>
    local label="$1"; shift
    [ -f "v1.3/results/score_$label.json" ] && { echo "AUTOSCORE skip $label (done)"; return; }
    echo "AUTOSCORE start $label $(date +%H:%M:%S)"
    # shellcheck disable=SC2086
    PYTHONPATH=src $PY scripts/report_v13.py "$@" > "v1.3/logs/score_$label.log" 2>&1
    echo "AUTOSCORE done $label exit=$? $(date +%H:%M:%S)"
}

while IFS=$'\t' read -r rung seed name; do
    # Wait for this run to finish. `saved` is the last line train() prints, so it is the only
    # signal that the checkpoint on disk is complete rather than half-written.
    until grep -q '^saved ' "v1.3/logs/train_$name.log" 2>/dev/null; do
        if grep -q 'Traceback' "v1.3/logs/train_$name.log" 2>/dev/null; then
            echo "AUTOSCORE $name FAILED to train -- skipping"; continue 2
        fi
        sleep "$POLL"
    done
    score "$name" --run "$name"
    score "${name}_e2e_refit" --run "$name" --motion predicted --smooth 9 --tol 0.5 --refit \
          --label "$name"
done < v1.3/logs/queue.plan
echo "AUTOSCORE all done $(date +%H:%M:%S)"
