"""Build one v2 dataset end to end: shots in, `datasets/v003` out.

The order is the order the charter puts it in, and each stage writes what the next one reads,
so a build is inspectable at every step rather than being one opaque pass:

1. **ingest** -- one ``.sfx`` per shot, by the ladder in :mod:`roto.v2.ingest`
2. **manifest** -- every top-level layer becomes an element, tagged not filtered
3. **pairing QC** -- charter S6's gate; grades each element gold or silver and can fail a shot
4. **build** -- render each element's alpha and derive its tensors
5. **splits** -- frame, element and shot holdouts recorded at build time

``qc_gates`` decides whether a shot failing step 3 is dropped or merely recorded. It defaults
to ``True`` because charter S6 says pairing QC "gates every ingested shot" -- but the failures
it catches (a collision, an orphaned delivered channel) say the *pairing* is wrong, not that
the roto is, so the flag exists for a round that wants the elements anyway with their grade
attached.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from ..dataset.crop import CropConfig
from . import subset as subset_mod
from .build import DATASET_VERSION, BuiltElement, build, v2_crop_config
from .ingest import find_shots
from .manifest import MANIFEST_VERSION, Element, elements
from .qc import QC_VERSION, ShotQC, check, grades
from .splits import DEFAULT_HOLDOUT_EVERY, build_splits


def build_dataset(data_root: str | Path, out_root: str | Path,
                  shots: Iterable[str] | None = None,
                  cfg: CropConfig | None = None,
                  held_shots: Iterable[str] | None = None,
                  holdout_every: int = DEFAULT_HOLDOUT_EVERY,
                  qc_gates: bool = True,
                  qc_scale: float = 0.4) -> dict[str, Any]:
    """Build every element of ``shots`` into ``out_root``. Returns the dataset manifest."""
    cfg = cfg or v2_crop_config()
    names = tuple(shots) if shots is not None else subset_mod.TIER1
    held = tuple(held_shots) if held_shots is not None else subset_mod.HOLDOUT_SHOTS

    found = find_shots(data_root, names)
    unusable = [s for s in found if not s.usable]
    usable = [s for s in found if s.usable]
    els = elements(usable)

    qc = check(usable, els, scale=qc_scale)
    graded = grades(qc)
    failed_shots = {n for n, r in qc.items() if not r.passed} if qc_gates else set()

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    by_shot = {s.name: s for s in usable}

    built: list[BuiltElement] = []
    skipped: list[dict[str, str]] = []
    for e in els:
        if e.shot in failed_shots:
            skipped.append({'element_id': e.element_id, 'why': 'shot failed pairing QC'})
            continue
        built.append(build(e, by_shot[e.shot], out, cfg, graded.get(e.element_id)))

    splits = build_splits(
        [(b.element_id, b.meta['element']['shot'], b.frames) for b in built],
        holdout_every=holdout_every,
        held_shots=[h for h in held if any(b.meta['element']['shot'] == h for b in built)],
        note='v2 Step 1. Shot holdout chosen in roto.v2.subset: one small, one dense.')
    (out / 'splits.json').write_text(json.dumps(splits, indent=2))

    manifest = {
        'dataset_version': DATASET_VERSION,
        'manifest_version': MANIFEST_VERSION,
        'qc_version': QC_VERSION,
        'data_root': str(data_root),
        'shots_requested': list(names),
        'shots_unusable': [s.as_dict() for s in unusable],
        'shots_failed_qc': sorted(failed_shots),
        'crop': asdict(cfg) if hasattr(cfg, '__dataclass_fields__') else {},
        'elements': [b.meta['element'] for b in built],
        'skipped': skipped,
        'grades': {eid: p.as_dict() for eid, p in graded.items()},
        'qc': {n: r.as_dict() for n, r in qc.items()},
        'counts': {
            'shots': len(usable), 'elements': len(built),
            'gold': sum(1 for b in built
                        if (b.meta.get('pairing') or {}).get('grade') == 'gold'),
            'silver': sum(1 for b in built
                          if (b.meta.get('pairing') or {}).get('grade') != 'gold'),
            'frames': sum(len(b.frames) for b in built),
            'shapes': sum(b.n_shapes for b in built),
            'trained_elements': len(splits['trained_elements']),
            'frames_held': splits['frames_held'],
        },
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(root: str | Path) -> dict[str, Any]:
    return json.loads((Path(root) / 'manifest.json').read_text())
