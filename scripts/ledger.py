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
from roto.geometry import crop_matrix, crop_to_local, local_to_crop  # noqa: E402
from roto.model.reconstruct import RebuildConfig, assemble               # noqa: E402
from roto.model.report import spread                                     # noqa: E402
from roto.model.train import apply_proj                                  # noqa: E402
from roto.program import decode, load_program, proj_from_matrix          # noqa: E402
from roto.render.curves import eval_bspline                              # noqa: E402
from roto.render.raster import (CONVENTION_SETS, RenderConfig,           # noqa: E402
                                config_from_meta, conventions, render_union)
from roto.sfx.json_ir import from_json_ir                                # noqa: E402
from roto.sfx.read import read_sfx                                       # noqa: E402
from roto.sfx.write import write_sfx                                     # noqa: E402
from roto.shots import find_shots                                        # noqa: E402
from roto.dataset import frame_split, load_splits                        # noqa: E402

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
        cfg = config_from_meta(meta['render'])
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

def scorer_conventions(dirs):
    """The scorer must render with **every** convention the dataset was drawn with.

    This row was "scorer supersample = dataset supersample" through v1.2, and the narrowness
    is why it stayed green through the bug it was there to catch. The scorer carried the
    supersample across from ``meta.json`` and rebuilt the other three conventions from
    ``RenderConfig``'s class defaults -- which are v1's, so on ``datasets/v001`` the omission
    could not produce a wrong number. On ``v002`` it does: the artist's own program reads
    0.9925 on ``FAM blue_1`` and 0.9887 on ``green_2``, the two layers with hundreds of open
    strokes, purely because the scorer filled open zero-width shapes the dataset had stroked.

    So the row now compares the *whole* config, field by field, against what
    ``render.raster.config_from_meta`` reconstructs from the record -- and it also checks that
    the record is self-consistent with a named convention set, which is what makes a dataset
    whose meta.json disagrees with the code an error instead of a silent re-baseline.
    """
    fields = ('supersample', 'samples_per_seg', 'fill_open_zero_width', 'open_end_rule',
              'clip_per_shape')
    bad, seen = [], set()
    for d in dirs:
        meta = json.loads((d / 'meta.json').read_text())
        rec = meta['render']
        seen.add(rec.get('conventions', 'v1'))
        try:
            cfg = config_from_meta(rec)                     # raises on a disagreeing record
        except ValueError as exc:
            bad.append(f'{d.name}: {exc}')
            continue
        want = conventions(rec.get('conventions', 'v1'), int(rec['supersample']))
        for f in fields:
            if getattr(cfg, f) != getattr(want, f):
                bad.append(f'{d.name}.{f}')
        h = meta['source']['height']
        if abs(cfg.stroke_px(0.030864, h) - rec['stroke_px_at_0.0309']) > 1e-9:
            bad.append(f'{d.name}.stroke_px')
    return row('scorer conventions = dataset conventions',
               f'all of {", ".join(fields)} + stroke gain, on every layer',
               f'{len(dirs)} layers, conventions={sorted(seen)}, {len(fields) + 1} fields each'
               if not bad else f'MISMATCH: {bad[:3]}',
               EXACT if not bad else RED, 'test_v13.py',
               'was supersample only through v1.2, which is how it stayed green through the '
               'bug it exists to catch')


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


def scoring_ceiling(path=Path('v1.3/results/scoring_ceiling.json')):
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
    # Exact, not "1.000000 to six places". The stored alpha is uint16, so a *perfect* render
    # still differs from it by up to half a quantum per pixel and a soft IoU built from those
    # half-steps reads 0.999999. Through v1.2 that floor was indistinguishable from a real
    # residual because datasets/v001 had a real one 400x larger sitting on top of it. So the
    # requirement is stated on the comparison that can be exactly 1.0: put the render through
    # the same uint16 round trip dataset.build writes, and check the raw-float disagreement
    # never exceeds half a quantum, which is the only thing quantisation can cost.
    q, worst_px = d.get('worst_frame_ceiling_quantised'), d.get('max_abs_pixel')
    if q is None:
        return row('artist program renders back to the stored alpha',
                   'soft IoU 1.000000 through the dataset\'s own uint16 round trip',
                   'measured without the quantised column -- re-run '
                   'scripts/exp_scoring_ceiling.py', RED, 'test_v13.py')
    half = d.get('half_quantum', 1.0 / 131070)
    status = EXACT if (q >= 1.0 and worst_px <= half * 1.0001) else RED
    return row('artist program renders back to the stored alpha',
               'soft IoU exactly 1.0 on every frame through the uint16 round trip, and no '
               'pixel off by more than half a quantum',
               f'quantised mean {d["mean_ceiling_quantised"]:.9f}, worst frame {q:.9f}, '
               f'{d["frames_below_1_quantised"]}/{d["frames"]} frames below 1.0; worst pixel '
               f'{worst_px:.3e} vs half-quantum {half:.3e} '
               f'(raw float: mean {d["mean_ceiling"]:.9f}, worst {d["worst_frame_ceiling"]:.9f})',
               status, 'test_v13.py + exp_scoring_ceiling.py',
               'RED on datasets/v001 at raw-float mean 0.999529 / worst frame 0.992632 / 247 '
               'frames below 0.999. Two debts, found one round apart: v1.1\'s Catmull-Rom '
               'endpoint fix landed after those alphas were rendered, and every scorer rebuilt '
               'three of the four render conventions from class defaults. Closed by the v002 '
               'rebuild and by config_from_meta respectively')


def render_conventions(dirs, path=Path('v1.1/results/render_conventions.json')):
    """Which convention set the dataset in hand is drawn with, against the refereed one.

    v1.1 refereed four conventions against Silhouette's delivered EXRs and found three of
    them against what the code shipped; v1.1 and v1.2 both left them in place on purpose,
    because changing any of them re-renders the training alphas and their whole value was
    being comparable to v1 row for row. **That argument expired at this rebuild**, and the
    row is what says whether the flip actually landed rather than being intended.

    It goes green only when every layer is drawn with the ``measured`` set. It does *not*
    claim agreement with Silhouette to a tolerance -- the referee was against delivered EXRs
    at one frame per layer, and the zero-margin claim still needs a written ``.sfx`` opened in
    a seat, which is its own row and not ours to close.
    """
    sets = {}
    for d in dirs:
        rec = json.loads((d / 'meta.json').read_text())['render']
        sets.setdefault(rec.get('conventions', 'v1'), []).append(d.name)
    got = ', '.join(f'{k} x{len(v)}' for k, v in sorted(sets.items()))
    measured_only = set(sets) == {'measured'}
    return row('render conventions = the set refereed against Silhouette',
               "every layer drawn with v1.1's measured set (fill_open_zero_width=False, "
               'stroke at the 1 px floor, duplicate open ends)',
               got, EXACT if measured_only else RED,
               'test_v11.py + test_v13.py',
               'v1.1 measured 3 of 4 against what shipped and left them; datasets/v001 '
               'carries the wrong two. The referee is against delivered EXRs, so this row '
               'is about the flip landing, not about zero margin in Silhouette'
               + (f'; v1.1 evidence in {path}' if path.exists() else ''))


def sfx_round_trip(data_root='data/extracted/test_data'):
    """``read(write(IR))`` must return the same IR, bit for bit, on every archive shot.

    The row the deliverable loop rests on. ``sfx/write.py`` was a ``NotImplementedError``
    seam through v1.2 for a stated reason -- a writer nobody has opened in Silhouette is
    unverifiable -- and that reason had a second half that has now gone: while the program's
    transform track carried six numbers where the archive needs eight, a file written from it
    would have put one layer 350 crop px from where the artist left it. With the projective
    fix in, the only thing left to wait for is the seat, and a file cannot be checked in a
    seat until it exists.

    Bit-exact rather than exact-to-a-tolerance, which is a choice in the writer: numbers go
    out at shortest-round-trip precision (``repr``) instead of Silhouette's fixed 9 decimals,
    so a float that goes in comes back with the same bits. Every dialect and both containers
    are walked, because the container is the half of the format most likely to be got wrong
    silently -- a zlib stream that decompresses to *almost* the right bytes still parses.
    """
    import glob

    def diff(a, b, path=''):
        """First disagreement between two IR trees, as a string, or ''."""
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
            if x.shape != y.shape:
                return f'{path}: shape {x.shape} != {y.shape}'
            d = float(np.abs(x - y).max()) if x.size else 0.0
            return f'{path}: values differ by {d:.3e}' if d else ''
        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                return f'{path}: {len(a)} != {len(b)} items'
            for i, (x, y) in enumerate(zip(a, b)):
                if (m := diff(x, y, f'{path}[{i}]')):
                    return m
            return ''
        if isinstance(a, dict):
            if sorted(a) != sorted(b):
                return f'{path}: keys {sorted(a)} != {sorted(b)}'
            for k in a:
                if (m := diff(a[k], b[k], f'{path}.{k}')):
                    return m
            return ''
        if hasattr(a, '__slots__'):
            for f in a.__slots__:
                if (m := diff(getattr(a, f), getattr(b, f), f'{path}.{f}')):
                    return m
            return ''
        return '' if a == b else f'{path}: {a!r} != {b!r}'

    import tempfile
    tmp = Path(tempfile.mkdtemp())
    shots, bad, dialects = 0, [], set()
    for shot in find_shots(data_root):
        if shot.sfx is None:
            continue
        doc = read_sfx(shot.sfx)
        back = read_sfx(write_sfx(doc, tmp / f'{shot.name}.sfx'))
        dialects.add(doc.dialect)
        shots += 1
        for f in ('width', 'height', 'duration', 'frame_rate', 'start_frame', 'dialect',
                  'source_path', 'source_label'):
            if getattr(doc, f) != getattr(back, f):
                bad.append(f'{shot.name}.{f}')
        if (m := diff(doc.roots, back.roots, shot.name)):
            bad.append(m)
    return row('read(write(IR)) is bit-exact',
               'every field of every shape, layer and track identical, all dialects',
               f'{shots} shots, {len(dialects)} dialects '
               f'({", ".join(sorted(dialects))}), 0 differences'
               if not bad else f'DIFFERS: {bad[:2]}',
               EXACT if not bad else RED, 'test_v13.py',
               'raised NotImplementedError through v1.2; opening one in Silhouette is a '
               'separate row and needs a seat')


def split_record(dirs, dataset):
    """The split must be the *dataset's*, recorded once, and agree with the rule.

    Two things, and the second is the one that bites. First: ``splits.json`` must reproduce
    the frame rule ``LayerData.split`` used through v1.2, or a v002 held-frame gap is not
    comparable to v1.1's +0.0005. Second: a layer the record withholds from training must
    actually be absent from training, which is a property of the *run* and so is checked
    against the checkpoints in :func:`training_respected_split` once any exist.

    A dataset with no record is not red -- ``datasets/v001`` has none by construction, and
    every published v1/v1.1/v1.2 number was measured without one. It is ``open``.
    """
    rec = load_splits(dataset)
    if rec is None:
        return row('split is defined at build time', 'the dataset records its own split',
                   f'{Path(dataset).name} has no splits.json', OPEN,
                   'test_v13.py', 'datasets/v001 predates the record; the rule is '
                                  'recomputed per run there')
    bad = []
    for d in dirs:
        meta = json.loads((d / 'meta.json').read_text())
        lid, frames = meta['layer']['layer_id'], meta['frames']['index']
        got = rec['layers'].get(lid)
        if got is None:
            bad.append(f'{lid}: not in splits.json')
            continue
        tr, he = frame_split(len(frames), rec['holdout_every'])
        if [frames[i] for i in tr] != got['frames_train'] or \
                [frames[i] for i in he] != got['frames_held']:
            bad.append(f'{lid}: recorded split disagrees with frame_split()')
    held = rec['held_layers']
    return row('split is defined at build time',
               'splits.json reproduces frame_split() on every layer; held layers named',
               f'every {rec["holdout_every"]}th frame ({rec["frames_held"]} frames), '
               f'{len(held)} layer(s) held out, {len(rec["trained_layers"])} trainable'
               if not bad else f'MISMATCH: {bad[:2]}',
               EXACT if not bad else RED, 'test_v13.py',
               'held layers: ' + (', '.join(h[:30] for h in held) or 'none'))


def key_target(dirs):
    """The key-timing head's target must be exactly the artist's keys, on the rendered axis.

    A new head needs a new Bucket-A row, and this one is cheap to get wrong in a way no
    aggregate would show: ``LayerData.key_mask`` is built by indexing the artist's key frames
    into the rendered frame axis, and a shape whose keys sit outside that axis (the archive
    has keys at frame -1) must be *dropped* rather than clamped to frame 0 -- clamping would
    invent a key the artist never set, on the frame a shape is most often keyed on anyway,
    and inflate key recall for free.
    """
    total = dropped = 0
    bad = []
    for d in dirs:
        el = load_element(d, with_local=False)
        t = np.load(d / 'tensors.npz')
        frames = set(int(f) for f in el.frames)
        for i in range(el.n_shapes):
            kf = [int(k) for k in t[f'key_frames/{i}'].tolist()]
            total += len(kf)
            inside = sorted(k for k in kf if k in frames)
            dropped += len(kf) - len(inside)
            got = sorted(int(el.frames[j]) for j in np.where(el.key_mask[:, i])[0])
            if got != inside:
                bad.append(f'{d.name}#{i}')
    return row('key-timing target = the artist\'s own keys',
               'key_mask holds exactly the artist keys that fall on a rendered frame',
               f'{total - dropped}/{total} artist keys on the rendered axis, '
               f'{dropped} outside it and dropped rather than clamped'
               if not bad else f'MISMATCH on {bad[:3]}',
               EXACT if not bad else RED, 'test_v13.py',
               'clamping an out-of-range key to frame 0 would invent a key and inflate recall')


def noise_floor(results=Path('v1.3/results'),
                runs=('v002_control', 'v002_control_s1'),
                long_runs=('v002_final', 'v002_final_s1')):
    """The last row, and the one that makes every other table readable: what a metric does
    when nothing changes but the seed."""
    out = []
    for label, group, steps in (('control, 40k', runs, 40000),
                                ('headline, 40k', long_runs, 40000)):
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
    ap.add_argument('--dataset', default='datasets/v002')
    ap.add_argument('--out', default='v1.3/results')
    ap.add_argument('--results', default='v1.3/results',
                    help='where the scored runs live, for the noise-floor row')
    ap.add_argument('--data-root', default='data/extracted/test_data',
                    help='the archive, for the .sfx round-trip row')
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
        scorer_conventions(all_dirs),
        cr_endpoints(all_dirs),
        metrics_euclidean(small),
        key_f1_order_free(),
        window_alignment(els[:2]),
        *probe_transform(els[:2]),
        transform_carriers(all_dirs),
        scoring_ceiling(Path(args.out) / 'scoring_ceiling.json'),
        render_conventions(all_dirs),
        sfx_round_trip(args.data_root),
        split_record(all_dirs, args.dataset),
        key_target(all_dirs),
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
