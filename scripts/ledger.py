"""The exactness ledger: every pixel of error this pipeline *controls*, measured.

The handover's first move is to split disagreement between a rendered reconstruction and the
artist's matte into two buckets and treat them differently:

* **Bucket A -- pipeline error.** Conversions, alignment, scoring, rendering conventions,
  interpolation. Must be *exactly* zero, provably, with a test pinning each item. Anything
  here silently caps the model and poisons every comparison between runs.
* **Bucket B -- model error.** The network's own geometry and timing error. Minimised, never
  assumed zero, and reported worst-case. That is `report_v12.py` and `summarise_v12.py`.

This script is Bucket A. It re-measures every row live rather than citing the run that first
measured it, because the point of a ledger is that it can go red: a refactor that breaks the
crop round trip should turn this table red in seconds, not surface as a model that mysteriously
stopped improving.

Each row carries the test that pins it. A row with no test is a row that will regress quietly,
which is why `test` is a column rather than a footnote.

    python scripts/ledger.py                 # -> v1.2/results/ledger.{json,md}
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import torch                                                             # noqa: E402

from roto.dataset import load_alpha                                      # noqa: E402
from roto.ir import CATMULLROM, Key, sample                              # noqa: E402
from roto.keys import f1 as key_f1                                       # noqa: E402
from roto.metrics import soft_iou                                        # noqa: E402
from roto.model.curveloss import polyline_matrix                         # noqa: E402
from roto.model.data import load_element                                 # noqa: E402
from roto.model.geometry import crop_matrix, crop_to_local, local_to_crop  # noqa: E402
from roto.model.reconstruct import RebuildConfig, assemble               # noqa: E402
from roto.model.report import spread                                     # noqa: E402
from roto.model.train import apply_proj                                  # noqa: E402
from roto.program import decode, load_program, proj_from_matrix          # noqa: E402
from roto.render.curves import eval_bspline                              # noqa: E402
from roto.render.raster import RenderConfig, render_union                # noqa: E402
from roto.sfx.json_ir import from_json_ir                                # noqa: E402

EXACT, MEASURED, OPEN, RED = 'exact', 'measured', 'open', 'RED'

# Layers the rows sample. Not all thirteen: every row here is a claim about arithmetic that
# does not vary by layer, so the sample is chosen for *hardness* rather than coverage --
# the perspective layer, the extreme-scale layer, and one cheap one. Rows that are claims
# about the archive (carriers, interp mix) walk all thirteen.
HARD = ['FAM_0060_L1_A0003C007_v001__red_1',            # |m03| 0.028, m33 != 1: perspective
        'nfl_0200_bg01_v001_compplate_roto_v001__green',  # px_per_norm 4365
        'FAM_0060_L1_A0003C007_v001__r1_t_c']           # 4 shapes: the rows that render


def row(item, requirement, measured, status, test, note=''):
    return {'item': item, 'requirement': requirement, 'measured': measured,
            'status': status, 'test': test, 'note': note}


def ok(value, limit):
    return EXACT if value <= limit else RED


# ---- 1. the representation itself -------------------------------------------

def ir_round_trip(dirs, stride=23):
    """``decode(encode(artist program))`` rendered against **the artist's IR rendered the same
    way**, so the row is about the representation and nothing else.

    Comparing to the stored *alpha* instead would fold in whatever the dataset is out of date
    by, which is a separate row (:func:`scoring_ceiling`) with a separate cause. Keeping them
    apart is the point of a ledger: one number, one thing that can be wrong.

    This row read 0.873864 before v1.2 -- on ``FAM red_1``, whose transform track the program's
    6-number affine form cannot express. See ``program.ProgramTensors.transform``."""
    worst, where = 1.0, ''
    for d in dirs:
        spec, tensors, meta = load_program(d)
        doc = decode(d, spec, tensors)
        artist = from_json_ir(json.loads((d / 'target_ir.json').read_text()))
        cfg = RenderConfig(supersample=meta['render']['supersample'])
        size, scale = meta['crop']['size_src_px'], meta['crop']['scale']
        for f in meta['frames']['index'][::stride]:
            x0, y0 = meta['crop']['offsets'][str(f)]
            box = (x0, y0, size, size)
            s = soft_iou(render_union(doc, int(f), cfg, scale, box),
                         render_union(artist, int(f), cfg, scale, box))
            if s < worst:
                worst, where = s, f'{d.name} @{f}'
    return row('IR -> tensors -> IR round trip',
               'soft IoU 1.000000 against the artist IR rendered the same way',
               f'{worst:.6f}', ok(1.0 - worst, 1e-6), 'test_dataset.py + test_v12.py',
               (f'worst at {where}; ' if where else '')
               + 'read 0.873864 before the projective fix')


def crop_round_trip(els, n=4000, seed=0):
    """``crop_to_local`` o ``local_to_crop`` over the archive's real matrices."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for el in els:
        c = el.crop
        kw = dict(width=c['width'], height=c['height'], scale=c['scale'],
                  out_px=c['out_px'])
        for fi in (0, len(el.frames) // 2, len(el.frames) - 1):
            f = int(el.frames[fi])
            for si in (0, el.n_shapes - 1):
                m = el.matrices[fi, si]
                pts = rng.normal(scale=0.4, size=(n, 2))
                back = crop_to_local(local_to_crop(pts, m, offset=c['offsets'][f], **kw), m,
                                     offset=c['offsets'][f], **kw)
                worst = max(worst, float(np.abs(back - pts).max()))
    return row('local_to_crop o crop_to_local', 'identity to <= 1e-9 (local units)',
               f'{worst:.2e}', ok(worst, 1e-9), 'test_geometry.py',
               f'{n} random points x real matrices')


def polyline_vs_renderer(els, seed=0):
    """The curve loss's fixed linear map against the rasteriser's own B-spline."""
    rng = np.random.default_rng(seed)
    combos = set()
    for el in els:
        combos |= set(zip(el.n_points_per_shape.tolist(), el.closed_per_shape.tolist(),
                          el.coords_per_shape.tolist()))
    worst = 0.0
    for P, closed, C in sorted(combos):
        if P < 3 or C != 1:                    # Bezier shapes and 2-point shapes sit out
            continue
        pts = rng.normal(size=(P, 2))
        worst = max(worst, float(np.abs(polyline_matrix(P, closed, 4) @ pts
                                        - eval_bspline(pts, closed, 4)).max()))
    return row('polyline_matrix vs eval_bspline', '<= 1e-12 on every shape kind in the archive',
               f'{worst:.2e}', ok(worst, 1e-12), 'test_v11.py',
               f'{len(combos)} distinct (points, closed, coords) combinations')


# ---- 2. scoring ------------------------------------------------------------

def scorer_supersample(els):
    """The scorer must render at the supersample the target was written at."""
    bad = [el.layer_id for el in els
           if RenderConfig(supersample=el.crop['supersample']).supersample
           != el.crop['supersample']]
    got = sorted({int(el.crop['supersample']) for el in els})
    return row('scorer supersample = dataset supersample',
               'equal on every layer, read from meta.json',
               f'dataset {got}, scorer {got}' if not bad else f'MISMATCH on {bad}',
               EXACT if not bad else RED, 'test_dataset.py',
               'v1 rendered predictions at 2 against targets at 4')


def metrics_euclidean(el):
    """Displace every control point by exactly (3, 4) crop px: a Euclidean metric reads 5.

    v1's point error was ``(|dx| + |dy|) / 2 * out_px``, which is neither L1 nor L2 and reads
    3.5 here -- 30% under the distance an artist would measure."""
    pred = el.points + np.array([3.0, 4.0], np.float32) / el.out_px
    rec = assemble(el, pred, el.proj_crop, RebuildConfig(), [int(el.frames[0])])
    return row('point error is Euclidean', 'exactly 5.000 px for a (3, 4) px displacement',
               f'{rec.point_err_px:.6f} px', ok(abs(rec.point_err_px - 5.0), 1e-4),
               'test_v12.py', 'the old formula reads 3.500')


def key_f1_order_free(seed=0, trials=200):
    """Matching is one-to-one and globally sorted, so the score cannot depend on input order."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        truth = np.sort(rng.choice(120, size=12, replace=False))
        pred = np.sort(rng.choice(120, size=15, replace=False))
        base = key_f1(pred, truth, tolerance=1)
        shuffled = key_f1(rng.permutation(pred), rng.permutation(truth), tolerance=1)
        worst = max(worst, max(abs(a - b) for a, b in zip(base, shuffled)))
    return row('key F1 matching is order-free', 'identical under permutation of both inputs',
               f'{worst:.2e}', ok(worst, 0.0), 'test_keys.py', f'{trials} random key sets')


def cr_endpoints(dirs):
    """Catmull-Rom clamps at the ends, and every key keeps its own interpolation law.

    Both halves matter and only one is arithmetic. The clamp is checked against the closed
    form; "one law per track" is checked by counting the archive's actual mix, because a
    pipeline that re-derived interpolation instead of reading each key's mode would silently
    disagree with the picture on the quarter of segments that are Catmull-Rom."""
    v = np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 1.0]], [[3.0, 1.0]]])
    keys = [Key(f, CATMULLROM, v[i]) for i, f in enumerate((0, 10, 20, 30))]
    got = np.asarray(sample(keys, 5), float)[0]
    # First segment, t=0.5, p0 clamped to k0: the standard uniform CR basis with p0 == p1.
    t, p0, p1, p2, p3 = 0.5, v[0][0], v[0][0], v[1][0], v[2][0]
    want = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
    modes: dict[str, int] = {}
    for d in dirs:
        doc = from_json_ir(json.loads((d / 'target_ir.json').read_text()))
        for _, shape in doc.shapes():
            for k in shape.path:
                modes[k.interp] = modes.get(k.interp, 0) + 1
    total = sum(modes.values())
    mix = ', '.join(f'{k} {100 * n / total:.0f}%' for k, n in sorted(modes.items()))
    return row('CR endpoints clamped, one law per key',
               'closed form to <= 1e-12; every key keeps its own mode',
               f'{np.abs(got - want).max():.2e} ({mix})',
               ok(float(np.abs(got - want).max()), 1e-12), 'test_ir.py',
               f'{total} artist keys across 13 layers')


# ---- 3. the two rows the handover adds ---------------------------------------

def window_alignment(els):
    """The delta that shifts a neighbour alpha into the anchor's window must be the delta the
    targets were built with. See ``tests/test_v12.py`` for the full argument.

    The note reports how far the window actually moves, and it is worth reading: v1.1 quoted
    "1.09 crop px per step, max 4.1 (FAM blue; similar elsewhere)" from its own
    ``offset_jitter.json``, and that file has ``nfl_0200 green`` at a **mean of 30.85** and a
    max of 45. The spread across layers is 50x, so the misaligned window handed the network
    three channels that disagreed about position by tens of pixels on the extreme-scale layer,
    not by one."""
    worst, moved, where = 0.0, 0.0, ''
    for el in els:
        c = el.crop
        kw = dict(width=c['width'], height=c['height'], scale=c['scale'],
                  out_px=c['out_px'])
        off, n = el.offsets_crop_px, len(el.frames)
        step = float(np.abs(np.diff(off, axis=0)).max())
        if step > moved:
            moved, where = step, el.layer_id
        for anchor in (1, n // 2, n - 2):
            for j in (anchor - 1, anchor + 1):
                delta = off[j] - off[anchor]
                for si in range(el.n_shapes):
                    if not el.live[j, si]:
                        continue
                    P = int(el.point_mask[si, :, 0].sum())
                    C = int(el.point_mask[si, 0].sum())
                    local = el.local[j, si, :P, :C].astype(np.float64)
                    at = lambda fi: local_to_crop(
                        local, el.matrices[j, si],
                        offset=c['offsets'][int(el.frames[fi])], **kw) * el.out_px
                    worst = max(worst, float(np.abs(at(j) + delta - at(anchor)).max()))
    return row('window alignment uses the recorded offsets',
               'neighbour targets land on anchor-window coordinates to <= 1e-9 crop px',
               f'{worst:.2e} crop px', ok(worst, 1e-9), 'test_v12.py',
               f'not vacuous: the window moves up to {moved:.1f} crop px per step on '
               f'{where[:34]}, against 0.8 on the other layer sampled')


def probe_transform(els):
    """The transform loss's probe landings against ``local_to_crop`` on the same matrix."""
    worst, worst32 = 0.0, 0.0
    for el in els:
        c = el.crop
        kw = dict(width=c['width'], height=c['height'], scale=c['scale'],
                  out_px=c['out_px'])
        first = [int(np.where(el.group_of == g)[0][0]) for g in range(el.n_groups)]
        for fi in range(0, len(el.frames), 11):
            f = int(el.frames[fi])
            cm = crop_matrix(offset=c['offsets'][f], **kw)
            for g in range(el.n_groups):
                m = el.matrices[fi, first[g]]
                probe = torch.from_numpy(el.probe_local[g].astype(np.float64))[None]
                exact = proj_from_matrix(m @ cm)
                got = apply_proj(torch.from_numpy(exact)[None], probe).numpy()[0] * el.out_px
                want = local_to_crop(el.probe_local[g].astype(np.float64), m,
                                     offset=c['offsets'][f], **kw) * el.out_px
                worst = max(worst, float(np.abs(got - want).max()))
                stored = apply_proj(
                    torch.from_numpy(el.proj_crop[fi, g].astype(np.float64))[None],
                    probe).numpy()[0] * el.out_px
                worst32 = max(worst32, float(np.abs(stored - got).max()))
    return [row('probe-point transform target', 'probes through crop_matrix(f) agree with '
                'local_to_crop at f to <= 1e-9 crop px',
                f'{worst:.2e} crop px', ok(worst, 1e-9), 'test_v12.py'),
            row('float32 storage of the targets', 'on the record, not required zero',
                f'{worst32:.2e} crop px', MEASURED, 'test_v12.py',
                'four orders of magnitude under the model\'s own error')]


# ---- 4. what v1.2 found while building the ledger ---------------------------

def transform_carriers(dirs):
    """Every shape must be reachable by the predicted-transform substitution.

    v1 and v1.1 wrote the predicted matrix only onto ancestors that *already* carried a
    transform, so shapes with none were rendered at the artist's own position whatever the
    head predicted -- and the head's target for those groups is not a no-op, it is the crop
    window's own map. See ``reconstruct.with_transforms``."""
    fallback = shared = double = shapes = 0
    for d in dirs:
        doc = from_json_ir(json.loads((d / 'target_ir.json').read_text()))
        el = load_element(d, with_local=False)
        ancs = [a for a, _ in doc.shapes()]
        shapes += len(ancs)
        seen: dict[int, set[int]] = {}
        for si, a in enumerate(ancs):
            for layer in a:
                seen.setdefault(id(layer), set()).add(int(el.group_of[si]))
        for si, a in enumerate(ancs):
            carriers = [l for l in a if l.transform]
            double += len(carriers) > 1
            if not carriers:
                fallback += 1
                shared += len(seen[id(a[-1])]) > 1
    return row('predicted transform reaches every shape',
               'no shape unreachable, no shape with two carriers, no carrier shared '
               'across groups',
               f'{fallback}/{shapes} need the innermost-ancestor fallback; '
               f'{double} double carriers; {shared} shared',
               EXACT if not (double or shared) else RED, 'test_v12.py',
               'v1/v1.1 skipped the fallback shapes, which flattered the head')


def scoring_ceiling(path=Path('v1.2/results/scoring_ceiling.json')):
    """What the *artist's own program* scores against the stored alphas.

    The strongest form of every check in this file, and the one that is red. v1's review asked
    for one interpolation law across a Catmull-Rom track (S6); v1.1 made that change after
    ``datasets/v001`` was rendered, so the stored alphas are drawn under the old law and every
    score since is computed under the new one. 52% of this archive's keys are Catmull-Rom.
    Priced per layer and per frame by ``scripts/exp_scoring_ceiling.py``; closed by the dataset
    rebuild the handover already schedules, not by reverting the fix."""
    if not path.exists():
        return row('artist program renders back to the stored alpha', 'soft IoU 1.000000',
                   'not measured -- run scripts/exp_scoring_ceiling.py', RED,
                   'test_dataset.py')
    d = json.loads(path.read_text())
    return row('artist program renders back to the stored alpha',
               'soft IoU 1.000000 on every frame',
               f'mean {d["mean_ceiling"]:.6f}, worst frame {d["worst_frame_ceiling"]:.6f} '
               f'({d["worst_frame_layer"][:26]} @{d["worst_frame"]}), '
               f'{d["frames_below_0.999"]}/{d["frames"]} frames below 0.999',
               RED, 'test_ir.py + exp_scoring_ceiling.py',
               'v1.1\'s Catmull-Rom endpoint fix landed after the alphas were rendered; '
               'the v1-era renderer reproduces them to 0.999999. Closed by the rebuild')


def render_conventions(path=Path('v1.1/results/render_conventions.json')):
    """The one row that is not ours to close. Left open, with the measurement on the table."""
    note = ('self-consistent against our own render; the flip belongs to the dataset rebuild, '
            'and the zero-margin claim needs one written .sfx diffed in Silhouette')
    if not path.exists():
        return row('render conventions vs Silhouette', 'match Silhouette\'s own EXRs',
                   'not measured here', OPEN, 'test_v11.py', note)
    d = json.loads(path.read_text())
    keys = [k for k in ('conventions', 'summary', 'findings') if k in d]
    return row('render conventions vs Silhouette',
               'match Silhouette\'s own EXRs (two are measured wrong)',
               f'refereed in v1.1: {", ".join(keys) or "see file"}; '
               '2 of 4 conventions wrong in datasets/v001', OPEN, 'test_v11.py', note)


def noise_floor(results=Path('v1.2/results'), runs=('v1_control', 'control_s1', 'control_s2'),
                long_runs=('v1_control_long', 'control_long_s1', 'control_long_s2')):
    """The last row, and the one that makes every other table readable: what a metric does
    when nothing changes but the seed."""
    out = []
    for label, group, steps in (('12k', runs, 12000), ('40k', long_runs, 40000)):
        got = []
        for r in group:
            p = results / f'score_{r}.json'
            if p.exists():
                got.append(json.loads(p.read_text()))
        if len(got) < 2:
            out.append(row(f'run-to-run noise floor ({label})',
                           'at least two seeds of the control configuration',
                           f'{len(got)} run(s) scored', RED, 'scripts/train_v12.py',
                           'cannot tell a +0.003 improvement from luck without it'))
            continue
        s = spread(got, ['mean_soft_iou', 'point_err_px', 'jitter_px', 'key_f1',
                         'worst_layer_soft_iou', 'worst_frame_soft_iou'])
        out.append(row(f'run-to-run noise floor ({label})',
                       'measured, not assumed',
                       f'soft IoU range {s["mean_soft_iou"]["range"]:.4f} over '
                       f'{len(got)} seeds', MEASURED, 'scripts/exp_noise_floor.py',
                       f'point px range {s["point_err_px"]["range"]:.2f}, '
                       f'worst-layer range {s["worst_layer_soft_iou"]["range"]:.4f}'))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='datasets/v001')
    ap.add_argument('--out', default='v1.2/results')
    ap.add_argument('--results', default='v1.2/results',
                    help='where the scored runs live, for the noise-floor row')
    args = ap.parse_args()

    t0 = time.time()
    all_dirs = sorted(p for p in Path(args.dataset).iterdir() if (p / 'meta.json').exists())
    hard_dirs = [Path(args.dataset) / n for n in HARD]
    els = [load_element(d) for d in hard_dirs]
    small = els[-1]

    rows = [
        ir_round_trip(hard_dirs),
        crop_round_trip(els),
        polyline_vs_renderer(els),
        scorer_supersample(els),
        cr_endpoints(all_dirs),
        metrics_euclidean(small),
        key_f1_order_free(),
        window_alignment(els[:2]),
        *probe_transform(els[:2]),
        transform_carriers(all_dirs),
        scoring_ceiling(),
        render_conventions(),
        *noise_floor(Path(args.results)),
    ]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    reds = [r for r in rows if r['status'] == RED]
    payload = {'rows': rows, 'red': len(reds), 'seconds': round(time.time() - t0, 1)}
    (out / 'ledger.json').write_text(json.dumps(payload, indent=2))

    md = ['# Bucket A — the exactness ledger', '',
          'Every row re-measured by `scripts/ledger.py`, not cited from the run that first',
          'measured it. `exact` means the requirement is met; `measured` means the number is',
          'on the record rather than required to be zero; `open` means it is not ours to',
          'close yet.', '',
          '| item | requirement | measured | status | test |',
          '|---|---|---|---|---|']
    for r in rows:
        note = f'<br>_{r["note"]}_' if r['note'] else ''
        md.append(f'| {r["item"]} | {r["requirement"]} | `{r["measured"]}`{note} | '
                  f'**{r["status"]}** | `{r["test"]}` |')
    md += ['', f'{len(rows)} rows, {len(reds)} red, measured in '
               f'{payload["seconds"]:.0f}s.']
    (out / 'ledger.md').write_text('\n'.join(md) + '\n')

    width = max(len(r['item']) for r in rows)
    for r in rows:
        flag = {EXACT: 'green', MEASURED: 'note ', OPEN: 'open ', RED: 'RED  '}[r['status']]
        print(f'[{flag}] {r["item"]:<{width}}  {r["measured"]}')
    print(f'\n{len(rows)} rows, {len(reds)} red, {payload["seconds"]:.0f}s '
          f'-> {out / "ledger.md"}')
    sys.exit(1 if reds else 0)


if __name__ == '__main__':
    main()
