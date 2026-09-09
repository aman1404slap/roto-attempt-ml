#!/bin/bash
# The v1.2 training queue, in the order the report needs the numbers.
#
# Sequential on purpose: one 8 GB GPU, and the point of the seed repeats is that nothing
# about the run varies except the seed -- sharing a card with another run would vary
# wall-clock and, with it, nothing the report claims, but it would also risk an OOM that
# invalidates a rung mid-ladder for no gain.
cd /home/aman/Projects/roto-v2 || exit 1
PY=~/.pyenv/shims/python3
mkdir -p v1.2/logs
for name in "$@"; do
    echo "QUEUE start $name $(date +%H:%M:%S)"
    $PY scripts/train_v12.py "$name" > "v1.2/logs/train_$name.log" 2>&1
    code=$?
    echo "QUEUE done $name exit=$code $(date +%H:%M:%S) :: $(grep -m1 '^saved' "v1.2/logs/train_$name.log")"
done
echo "QUEUE all done $(date +%H:%M:%S)"
