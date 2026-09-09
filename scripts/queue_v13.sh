#!/bin/bash
# The v1.3 training queue, in the order the report needs the numbers.
#
# Sequential on purpose: one 8 GB card, and the whole point of a seed repeat is that nothing
# varies except the seed. Sharing the card would vary wall clock -- which the report does not
# claim -- and risk an OOM that invalidates a rung mid-queue for no gain.
#
# Writes v1.3/logs/queue.json first so scripts/track_v13.py can report progress and an ETA for
# the queue as a whole rather than for whichever job happens to be running.
#
#   scripts/queue_v13.sh v002_control v002_control:1 v002_final v002_final:1 ...
#
# A ":N" suffix means "this rung at seed N", which is how the two-seed rows are asked for.
cd /home/aman/Projects/roto-v2 || exit 1
PY=~/.pyenv/shims/python3
mkdir -p v1.3/logs v1.3/runs

# One place resolves a spec to (rung, seed, run name, steps): the queue file for the tracker
# and a plan file for this loop, so the two cannot disagree about what is being run.
$PY - "$@" <<'PYEOF'
import json, sys
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
from train_v13 import LADDER
jobs = []
for spec in sys.argv[1:]:
    rung, _, seed = spec.partition(':')
    cfg = LADDER[rung]
    s = int(seed) if seed else cfg.seed
    # `self_attn` is recorded because it is what predicts throughput: O(S^2) at S=1036 is
    # the difference between 48 and 15 steps/s on this card, so an ETA that pools the two
    # is wrong by 3x. See scripts/track_v13.py.
    jobs.append({'name': f'{rung}_s{s}' if s != cfg.seed else rung, 'rung': rung,
                 'seed': s, 'steps': cfg.steps, 'self_attn': bool(cfg.self_attn)})
json.dump({'jobs': jobs}, open('v1.3/logs/queue.json', 'w'), indent=2)
with open('v1.3/logs/queue.plan', 'w') as fh:
    for j in jobs:
        fh.write(f'{j["rung"]}\t{j["seed"]}\t{j["name"]}\n')
PYEOF
[ -s v1.3/logs/queue.plan ] || { echo "queue: nothing to run"; exit 1; }

while IFS=$'\t' read -r rung seed name; do
    echo "QUEUE start $name $(date +%H:%M:%S)"
    $PY scripts/train_v13.py "$rung" --seed "$seed" > "v1.3/logs/train_$name.log" 2>&1
    echo "QUEUE done $name exit=$? $(date +%H:%M:%S) :: $(grep -m1 '^saved' "v1.3/logs/train_$name.log")"
done < v1.3/logs/queue.plan
echo "QUEUE all done $(date +%H:%M:%S)"
