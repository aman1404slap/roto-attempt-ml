"""Live progress and ETA for the v1.3 queues, read off the logs the jobs are already writing.

There is nothing to instrument: ``train()`` flushes a ``step N ... [Ts]`` line every
``log_every`` steps and ``report_v13.py`` flushes one line per layer, so a tracker only has to
read the tail of a log and do arithmetic. That is deliberate -- a tracker that needed the jobs
to cooperate would be one more thing to get wrong in a five-hour queue, and a queue that dies
silently at 3am is the failure this exists to prevent.

ETA comes from each run's *own* observed rate rather than from a table of expectations,
because throughput here is not a constant: measured across v1.2's rungs it ranges from 14.5 to
48.6 steps/s on the same card. A run that has not started yet borrows the mean rate of
finished runs **with the same ``self_attn`` setting**, which is the axis that predicts it --
self-attention is O(S^2) and S reaches 1036, so pooling the two would make a five-hour ETA
wrong by a factor of three. Borrowed rates are labelled ``est``.

    python scripts/track_v13.py                       # one snapshot
    python scripts/track_v13.py --watch 300           # every 5 minutes until the queue drains
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

STEP = re.compile(r'^\s*step\s+(\d+)\s.*\[(\d+)s\]')
"""``  step  2500  point   3.152px ...  total  4.021  [58s]`` -- what ``train()`` prints."""

SAVED = re.compile(r'^saved .*\((\S+)M params, (\d+)s')


def humanise(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return '   ?'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f'{h}h{m:02d}m' if h else f'{m:2d}m{s:02d}s'


def read_train(log: Path, steps: int) -> dict:
    """Progress of one training run from its stdout log."""
    out = {'kind': 'train', 'steps': steps, 'step': 0, 'elapsed': 0.0, 'done': False,
           'rate': 0.0, 'state': 'pending'}
    if not log.exists():
        return out
    text = log.read_text(errors='replace')
    for line in text.splitlines():
        m = STEP.match(line)
        if m:
            out['step'], out['elapsed'] = int(m.group(1)), float(m.group(2))
        m = SAVED.match(line)
        if m:
            out['done'], out['elapsed'] = True, float(m.group(2))
            out['step'] = steps
    if 'Traceback' in text:
        out['state'] = 'FAILED'
    elif out['done']:
        out['state'] = 'done'
    elif out['step']:
        out['state'] = 'running'
    else:
        out['state'] = 'starting'
    out['rate'] = out['step'] / out['elapsed'] if out['elapsed'] else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--queue', default='v1.3/logs/queue.json',
                    help='written by queue_v13.sh: the runs, in order, with their step counts')
    ap.add_argument('--logs', default='v1.3/logs')
    ap.add_argument('--watch', type=int, default=0, metavar='SECONDS',
                    help='re-print every N seconds until nothing is left running')
    args = ap.parse_args()

    while True:
        q = json.loads(Path(args.queue).read_text()) if Path(args.queue).exists() else {}
        jobs = q.get('jobs', [])
        rows, remaining = [], 0.0
        # Rate to assume for a run that has not started: the mean over finished runs, so the
        # estimate improves as the queue drains instead of being a constant that is wrong.
        seen: dict[bool, list[float]] = {True: [], False: []}
        for j in jobs:
            r = read_train(Path(args.logs) / f'train_{j["name"]}.log', j['steps'])
            if r['state'] == 'done' and r['rate']:
                seen[bool(j.get('self_attn'))].append(r['rate'])
            rows.append((j, r))
        # Measured on this card in v1.2; replaced by the observed mean as runs finish.
        prior = {True: 14.7, False: 45.0}
        rate_for = lambda sa: (sum(seen[sa]) / len(seen[sa]) if seen[sa] else prior[sa])

        print(f'\n=== v1.3 queue @ {time.strftime("%H:%M:%S")} '
              f'({sum(1 for _, r in rows if r["state"] == "done")}/{len(rows)} done) ===')
        for j, r in rows:
            pct = 100.0 * r['step'] / max(1, r['steps'])
            fallback = rate_for(bool(j.get('self_attn')))
            rate = r['rate'] or fallback
            left = max(0.0, (r['steps'] - r['step']) / rate) if rate else float('nan')
            if r['state'] != 'done':
                remaining += left
            bar = '#' * int(pct / 5) + '.' * (20 - int(pct / 5))
            note = (f'{r["rate"]:4.1f} st/s' if r['rate'] else
                    f'~{fallback:.1f} st/s est')
            print(f'  {j["name"]:<22} {r["state"]:<8} [{bar}] {pct:5.1f}%  '
                  f'{r["step"]:>6}/{r["steps"]:<6} {note}  '
                  f'elapsed {humanise(r["elapsed"])}  '
                  f'{"" if r["state"] == "done" else "left " + humanise(left)}')
        print(f'  {"TOTAL":<22} {"":<8} remaining {humanise(remaining)}   '
              f'ETA {time.strftime("%H:%M", time.localtime(time.time() + remaining))}',
              flush=True)

        live = [r for _, r in rows if r['state'] in ('pending', 'starting', 'running')]
        if not args.watch or not live:
            if not live and rows:
                bad = [j['name'] for j, r in rows if r['state'] == 'FAILED']
                print(f'\nqueue drained. {len(bad)} failed'
                      + (f': {", ".join(bad)}' if bad else ''), flush=True)
            return
        time.sleep(args.watch)


if __name__ == '__main__':
    main()
