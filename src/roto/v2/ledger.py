"""The exactness ledger for a v2 dataset: charter L3's Bucket A, one row per thing.

Charter L3 splits every pixel of disagreement in two. **Bucket A** -- conversions, alignment,
scoring, rendering, interpolation -- must be exactly zero, each item pinned by a row that fails
loudly. **Bucket B** is model error, minimised and reported worst case. This module is Bucket A
for the dataset, and it is the Step 1 gate in ``v2-tracker.md``.

One row, one thing that can be wrong. A row that could go red for two different reasons is a
row that cannot tell you which, so rows are kept narrow even where that means more of them.

These rows need no model and no checkpoint. Rows about a trained run belong to Step 2 and are
not here -- an empty ledger that only ever reports what it actually measured is worth more than
a full one carrying rows it inherited.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..metrics import soft_iou
from ..render.raster import config_from_meta, render_union
from ..sfx.json_ir import from_json_ir
from ..sfx.read import read_sfx, read_xml_bytes
from ..sfx.write import write_sfx
from .build import element_dirs, load_alpha, load_meta
from .dataset import load_manifest

GREEN, RED, INFO = 'GREEN', 'RED', 'INFO'


@dataclass(slots=True)
class Row:
    item: str
    requirement: str
    measured: str
    status: str
    note: str = ''

    def as_dict(self) -> dict[str, Any]:
        return {'item': self.item, 'requirement': self.requirement,
                'measured': self.measured, 'status': self.status, 'note': self.note}


def _ok(error: float, limit: float) -> str:
    return GREEN if error <= limit else RED


def scoring_ceiling(dirs: Sequence[Path], stride: int = 7) -> Row:
    """**The Step 1 gate.** The artist's own shapes, re-rendered, against the stored alpha.

    This is the number plan S1 names: "artist-self-score == 1.000000 on every layer, every
    frame". It asks whether a *perfect* answer would score a perfect number -- if it would not,
    every model number below it is part pipeline error and part model error with no way to tell
    them apart, and nothing downstream means anything.

    **Compared through the dataset's own uint16 round trip, and the distinction is the row.**
    The stored alpha is uint16; a fresh render is float. Comparing them raw can never reach
    1.0, and the residue is not error -- it is the quantiser. Measured here, the worst raw
    disagreement is 7.63e-06 against a uint16 step of 1.526e-05: exactly half a step, which is
    what correct rounding is *defined* to cost. A row demanding 1.0 from that comparison would
    be red forever while nothing was wrong, and a row that loosened its tolerance to let it pass
    would stop being able to see a real disagreement of the same size. So the fresh render is
    quantised the way the stored one was, the requirement stays exactly 1.0, and the raw float
    is reported beside it.
    """
    worst, where, worst_px, worst_raw = 1.0, '', 0.0, 1.0
    for d in dirs:
        meta = load_meta(d)
        artist = from_json_ir(json.loads((d / 'target_ir.json').read_text()))
        cfg = config_from_meta(meta['render'])
        size, scale = meta['crop']['size_src_px'], meta['crop']['scale']
        for f in meta['frames']['index'][::stride]:
            x0, y0 = meta['crop']['offsets'][str(f)]
            stored = load_alpha(d, f)
            fresh = render_union(artist, int(f), cfg, scale, (x0, y0, size, size))
            quantised = (np.round(np.clip(fresh, 0, 1) * 65535).astype(np.uint16)
                         .astype(np.float32) / 65535.0)
            s = soft_iou(quantised, stored)
            worst_raw = min(worst_raw, soft_iou(fresh, stored))
            worst_px = max(worst_px, float(np.abs(quantised - stored).max()))
            if s < worst:
                worst, where = s, f'{d.name} @{f}'
    return Row('artist shapes render back to the stored alpha',
               'soft IoU exactly 1.0 on every element, every sampled frame',
               f'{worst:.9f}', _ok(1.0 - worst, 0.0),
               (f'worst at {where}; ' if where else '')
               + f'max abs pixel {worst_px:.2e}; before quantisation {worst_raw:.9f} '
                 f'(uint16 step 1.526e-05)')


def alpha_quantisation(dirs: Sequence[Path], stride: int = 11) -> Row:
    """The stored PNG is a faithful uint16 of what was rendered, and nothing more.

    Separate from :func:`scoring_ceiling` because they fail for different reasons: this one
    goes red if the image round trip is lossy, that one if the *renderer* disagrees with itself.
    """
    worst, where = 0.0, ''
    for d in dirs:
        meta = load_meta(d)
        for f in meta['frames']['index'][::stride]:
            a = load_alpha(d, f)
            q = np.round(np.clip(a, 0, 1) * 65535).astype(np.uint16).astype(np.float32) / 65535.0
            e = float(np.abs(a - q).max())
            if e > worst:
                worst, where = e, f'{d.name} @{f}'
    return Row('stored alpha survives its own uint16 round trip',
               'max abs pixel error 0', f'{worst:.3e}', _ok(worst, 1e-9),
               f'worst at {where}' if where else '')


def render_conventions(dirs: Sequence[Path]) -> Row:
    """Every element records which conventions drew it, and the record verifies.

    ``config_from_meta`` raises when a dataset's recorded flags disagree with the convention set
    it names. That guard is the row: a convention living in a *default* is one two code paths
    can disagree about while both look right, which cost this project two rounds.
    """
    seen, bad = set(), []
    for d in dirs:
        meta = load_meta(d)
        try:
            cfg = config_from_meta(meta['render'])
        except ValueError as e:
            bad.append(f'{d.name}: {e}')
            continue
        seen.add((meta['render']['conventions'], cfg.fill_open_zero_width,
                  cfg.open_end_rule, cfg.clip_per_shape, cfg.samples_per_seg))
    if bad:
        return Row('render conventions travel with the data', 'record verifies against code',
                   f'{len(bad)} disagree', RED, bad[0])
    if len(seen) != 1:
        return Row('render conventions travel with the data', 'one convention set per dataset',
                   f'{len(seen)} distinct sets', RED, f'{sorted(seen)}')
    name, fill, ends, clip, sps = next(iter(seen))
    return Row('render conventions travel with the data',
               'one verified convention set across every element',
               f'{name}: fill_open_zero_width={fill}, open_end_rule={ends}, '
               f'clip_per_shape={clip}, samples_per_seg={sps}', GREEN,
               f'{len(dirs)} elements agree')


def frame_coverage(dirs: Sequence[Path]) -> Row:
    """Every frame the meta claims exists on disk, and the count matches the document."""
    missing, mismatched = [], []
    for d in dirs:
        meta = load_meta(d)
        idx = meta['frames']['index']
        expect = len(range(0, meta['source']['duration'], max(1, meta['frames']['stride'])))
        if len(idx) != expect:
            mismatched.append(f'{d.name}: {len(idx)} frames, document implies {expect}')
        for f in idx:
            if not (d / 'alpha' / f'{f:05d}.png').exists():
                missing.append(f'{d.name}@{f}')
    problems = mismatched + missing
    return Row('every declared frame is on disk', 'no missing alpha, no count mismatch',
               'complete' if not problems else f'{len(problems)} problems',
               GREEN if not problems else RED, problems[0] if problems else '')


def split_record(root: Path, dirs: Sequence[Path]) -> Row:
    """The dataset declares its own split, it covers every element, and it is disjoint."""
    splits = json.loads((root / 'splits.json').read_text())
    els = splits['elements']
    problems = []
    if {d.name for d in dirs} != set(els):
        problems.append('split record and built elements disagree')
    for eid, rec in els.items():
        if set(rec['frames_train']) & set(rec['frames_held']):
            problems.append(f'{eid}: train and held frames overlap')
        if len(rec['frames_train']) + len(rec['frames_held']) != rec['n_frames']:
            problems.append(f'{eid}: frames do not sum to n_frames')
    for shot in splits['held_shots']:
        if any(r['in_train'] for r in els.values() if r['shot'] == shot):
            problems.append(f'held shot {shot} still has a trained element')
    return Row('the dataset declares what a run may train on',
               'every element split, disjoint, held shots fully withheld',
               f"{len(splits['trained_elements'])} trained, "
               f"{len(els) - len(splits['trained_elements'])} held, "
               f"{splits['frames_held']} frames held"
               if not problems else f'{len(problems)} problems',
               GREEN if not problems else RED,
               problems[0] if problems else
               f"holdout every {splits['holdout_every']}th frame; "
               f"held shots {splits['held_shots']}")


def sfx_round_trip(manifest: dict[str, Any]) -> Row:
    """``read(write(doc))`` is bit-exact for every ``.sfx`` this dataset was built from.

    Extends the archive's coverage to dialect ``6/plain``, which appears in this delivery and
    is not in ``write.DIALECT_CONTAINER``; unknown versions fall back to a plain container,
    which is what 6 uses, so the row is what confirms the fallback is right rather than lucky.
    """
    paths, seen = [], set()
    for e in manifest['elements']:
        if e['sfx'] not in seen:
            seen.add(e['sfx'])
            paths.append(Path(e['sfx']))
    bad, dialects = [], set()
    for p in paths:
        doc = read_sfx(p)
        dialects.add(doc.dialect)
        try:
            again = _round_trip(doc)
        except Exception as exc:
            bad.append(f'{p.name}: {type(exc).__name__}: {exc}')
            continue
        if to_key(again) != to_key(doc):
            bad.append(f'{p.name}: round trip differs')
    return Row('read(write(doc)) is bit-exact',
               'every .sfx the dataset was built from',
               f'{len(paths) - len(bad)}/{len(paths)} exact',
               GREEN if not bad else RED,
               (bad[0] if bad else f'dialects {sorted(dialects)}'))


def _round_trip(doc):
    """``read(write(doc))`` through a temporary file, in the document's own dialect."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        return read_sfx(write_sfx(doc, Path(tmp) / 'rt.sfx', doc.dialect))


def to_key(doc) -> str:
    from ..sfx.json_ir import to_json_ir
    return json.dumps(to_json_ir(doc), sort_keys=True)


def pairing_qc(manifest: dict[str, Any]) -> Row:
    """Charter S6: pairing QC gated every ingested shot, and the grades are recorded."""
    qc = manifest['qc']
    failed = sorted(n for n, r in qc.items() if not r['passed'])
    counts = manifest['counts']
    golds = [p['score'] for p in manifest['grades'].values() if p['grade'] == 'gold']
    return Row('pairing QC gates every ingested shot',
               'no shot ingested with an ambiguous pairing',
               f"{len(qc) - len(failed)}/{len(qc)} shots pass; "
               f"{counts['gold']} gold, {counts['silver']} silver",
               GREEN if not failed else RED,
               (f'failed: {failed}' if failed else
                f'gold agreement with the artist EXRs: '
                f'{min(golds):.4f}-{max(golds):.4f}' if golds else 'no gold pairings'))


def build_ledger(root: str | Path) -> list[Row]:
    root = Path(root)
    dirs = element_dirs(root)
    manifest = load_manifest(root)
    return [
        scoring_ceiling(dirs),
        alpha_quantisation(dirs),
        render_conventions(dirs),
        frame_coverage(dirs),
        split_record(root, dirs),
        sfx_round_trip(manifest),
        pairing_qc(manifest),
    ]


def render(rows: Sequence[Row]) -> str:
    w = max(len(r.item) for r in rows)
    out = []
    for r in rows:
        out.append(f'{r.status:5s}  {r.item:{w}s}  {r.measured}')
        if r.note:
            out.append(f'{"":5s}  {"":{w}s}  {r.note}')
    reds = sum(1 for r in rows if r.status == RED)
    out.append('')
    out.append(f'{len(rows) - reds}/{len(rows)} green' if not reds
               else f'{reds} RED of {len(rows)}')
    return '\n'.join(out)
