"""Referee the unverified render conventions against Silhouette's own delivered EXRs.

Four rules in the renderer were calibrated by eye rather than derived, and between them they
decide the geometry of every open-stroke shape -- 51% of the archive:

    1. ``curves.OPEN_END_RULE``     how an open B-spline extends past its end points
    2. ``RenderConfig.fill_open_zero_width``  does a zero-width open shape fill
    3. ``raster.STROKE_WIDTH_GAIN`` the strokeWidth -> pixels unit, suspiciously 1/16
    4. ``RenderConfig.clip_per_shape``  clip after each shape, or once at layer output

None of them needs to stay a guess. Each shot ships Silhouette's render of its own mattes, so
the experiment is: render the artist's shapes both ways and see which agrees with the
reference. Scored at **full resolution**, because these conventions differ only at the edge
and a downscaled compare hides exactly that.

Layers are chosen for signal, not for membership of the training set: a convention governing
open strokes can only be refereed where open strokes dominate. Two shots are used for the
stroke-width test on purpose -- if 1/16 is a *unit conversion* the same value wins on both,
and if it is a per-shot knob it will not.

    python scripts/exp_conventions.py --out v1.1/results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from roto.dataset.layers import layer_doc, top_layers                   # noqa: E402
from roto.exr import best_channel, matte_sequences, score_against_channel  # noqa: E402
from roto.render.curves import OPEN_END_RULES                           # noqa: E402
from roto.render.raster import RenderConfig                             # noqa: E402
from roto.sfx.read import read_sfx                                      # noqa: E402
from roto.shots import find_shot                                        # noqa: E402

DATA = 'data/extracted/test_data'

# (shot, layer, why this layer can referee anything)
TARGETS = [
    ('FAM_0060_L1_A0003C007_v001', 'green 2',
     '216 of 269 shapes open, strokeWidth 0.0309 -- end rule and stroke width'),
    ('nfl_0080_bg02_v001_compplate_roto_v001', 'MB 2',
     '508 of 510 shapes open at strokeWidth 0.0 -- the zero-width fill rule'),
    ('TVC_SHOTS_sh0230_BG01_v003_roto_v02', 'L110',
     '2515 of 2515 open, widths 0.039/0.0625/0.078 -- second shot for the 1/16 test'),
    ('FAM_0060_L1_A0003C007_v001', 'blue 1',
     '7 Subtract shapes among 1036 Adds -- the clip-ordering rule'),
]

GAINS = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
"""Stroke-width gains to sweep. 0.0625 = 1/16 is the current value; the powers of two above
and below it are what would distinguish 'the unit is 1/16' from 'the number was tuned'.

The range runs down to 1/256 because the first sweep improved monotonically all the way to
its own floor, which identifies nothing. A sweep has to contain its optimum to be evidence.
Note ``default_stroke_px`` floors at 1 px, so past some gain every stroke is a hairline and
the curve must flatten -- where it flattens is itself the answer."""

CONTROLS = [
    ('nfl_0200_bg01_v001_compplate_roto_v001', 'blue', '75 shapes, none open'),
    ('FAM_0060_L1_A0003C007_v001', 'green', '147 shapes, none open'),
    ('FAM_0060_L1_A0003C007_v001', 'blue', '11 shapes, none open'),
]
"""Layers with **no open shapes at all**, so none of the four conventions can apply to them.

These exist to separate two explanations of the same evidence. Every convention change that
reduces coverage improved agreement on the stroke layers, which is either (a) those
conventions being wrong, or (b) the renderer over-covering everywhere for an unrelated
reason. If closed-only layers already agree with Silhouette to ~0.99 then (b) is ruled out
and the stroke findings stand on their own. If they do not, the stroke sweep was measuring
the global bias and says nothing about strokes."""


def gain_fn(gain: float):
    return lambda width, height: max(1.0, width * height * gain)


def sweep(doc, seqs, matte, channel, frames, variants, scale):
    rows = []
    for label, cfg in variants:
        t = time.time()
        s = score_against_channel(doc, seqs, matte, channel, frames, cfg, scale)
        rows.append({'variant': label, 'mean_soft_iou': float(np.mean(s)) if s else None,
                     'frames': len(s), 'seconds': round(time.time() - t, 1)})
        print(f'    {label:<34} soft IoU {rows[-1]["mean_soft_iou"]:.4f}  '
              f'[{rows[-1]["seconds"]:.0f}s]')
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='v1.1/results')
    ap.add_argument('--scale', type=float, default=1.0, help='1.0 = full resolution')
    ap.add_argument('--frames', type=int, default=5, help='sample frames per layer')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = {'scale': args.scale, 'frames_per_layer': args.frames, 'targets': [],
              'controls': []}
    for shot_name, layer_name, why in CONTROLS:
        shot = find_shot(f'{DATA}/{shot_name}')
        doc = read_sfx(shot.sfx, validate=False)
        ref = next((r for r in top_layers(doc) if r.name == layer_name), None)
        if ref is None:
            print(f'!! {shot_name}/{layer_name}: no such layer'); continue
        sub = layer_doc(doc, ref)
        seqs = matte_sequences(shot.path)
        probe = [int(f) for f in np.linspace(0, doc.duration - 1, 3).astype(int)]
        matte, channel, quality = best_channel(sub, seqs, probe, RenderConfig(supersample=1),
                                               scale=0.35)
        frames = [int(f) for f in np.linspace(doc.duration * 0.15, doc.duration * 0.85,
                                              args.frames).astype(int)]
        print(f'\n=== CONTROL {shot_name} / {layer_name} ({why}) ===')
        print(f'    matched {matte}/{channel} at soft IoU {quality:.4f}')
        rows = sweep(sub, seqs, matte, channel, frames,
                     [('as shipped (ss=2)', RenderConfig(supersample=2)),
                      ('ss=4', RenderConfig(supersample=4))], args.scale)
        report['controls'].append({'shot': shot_name, 'layer': layer_name, 'why': why,
                                   'matte': matte, 'channel': channel,
                                   'match_quality': quality, 'variants': rows})
        (out / 'render_conventions.json').write_text(json.dumps(report, indent=2))

    for shot_name, layer_name, why in TARGETS:
        shot = find_shot(f'{DATA}/{shot_name}')
        doc = read_sfx(shot.sfx, validate=False)
        ref = next((r for r in top_layers(doc) if r.name == layer_name), None)
        if ref is None:
            print(f'!! {shot_name}/{layer_name}: no such layer'); continue
        sub = layer_doc(doc, ref)
        seqs = matte_sequences(shot.path)
        shapes = [s for _, s in sub.shapes()]
        opens = [s for s in shapes if not s.closed]

        print(f'\n=== {shot_name} / {layer_name} ===')
        print(f'    {why}')
        print(f'    {doc.width}x{doc.height}, {len(shapes)} shapes, {len(opens)} open, '
              f'widths {sorted({round(s.stroke_width, 5) for s in opens})}')

        # Which delivered channel does this layer explain? Coarse scale: it is a coarse
        # question, and only the referee itself needs full resolution.
        probe = [int(f) for f in np.linspace(0, doc.duration - 1, 3).astype(int)]
        matte, channel, quality = best_channel(sub, seqs, probe, RenderConfig(supersample=1),
                                               scale=0.35)
        print(f'    matched {matte}/{channel} at soft IoU {quality:.4f} (match quality)')
        frames = [int(f) for f in
                  np.linspace(doc.duration * 0.15, doc.duration * 0.85, args.frames).astype(int)]

        entry = {'shot': shot_name, 'layer': layer_name, 'why': why,
                 'width': doc.width, 'height': doc.height,
                 'shapes': len(shapes), 'open': len(opens),
                 'stroke_widths': sorted({round(s.stroke_width, 6) for s in opens}),
                 'matte': matte, 'channel': channel, 'match_quality': quality,
                 'frames': frames, 'sweeps': {}}

        if opens:
            print('  open end rule:')
            entry['sweeps']['open_end_rule'] = sweep(
                sub, seqs, matte, channel, frames,
                [(r, RenderConfig(supersample=2, open_end_rule=r)) for r in OPEN_END_RULES],
                args.scale)

            zero_width = any(s.stroke_width == 0 for s in opens)
            if zero_width:
                print('  zero-width open shapes:')
                entry['sweeps']['fill_open_zero_width'] = sweep(
                    sub, seqs, matte, channel, frames,
                    [(f'fill={v}', RenderConfig(supersample=2, fill_open_zero_width=v))
                     for v in (True, False)], args.scale)
            if any(s.stroke_width > 0 for s in opens):
                print('  stroke width gain:')
                entry['sweeps']['stroke_width_gain'] = sweep(
                    sub, seqs, matte, channel, frames,
                    [(f'gain={g:g} (1/{1/g:g})',
                      RenderConfig(supersample=2, fill_open_zero_width=False,
                                   stroke_px=gain_fn(g))) for g in GAINS], args.scale)

        if any(s.blend == 'Subtract' for s in shapes):
            print('  clip ordering:')
            entry['sweeps']['clip_per_shape'] = sweep(
                sub, seqs, matte, channel, frames,
                [(f'clip_per_shape={v}', RenderConfig(supersample=2, clip_per_shape=v))
                 for v in (True, False)], args.scale)

        report['targets'].append(entry)
        (out / 'render_conventions.json').write_text(json.dumps(report, indent=2))
    print(f'\nwrote {out / "render_conventions.json"}')


if __name__ == '__main__':
    main()
