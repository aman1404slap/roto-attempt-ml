"""v1 is deleted, and this is what keeps it deleted.

The boundary was written so v1 could be removed without touching v2, and it worked: the
removal was a delete, not an untangling. The test outlives its original job. A docstring
saying "v1 is gone" is a wish, and a module re-added in a hurry six weeks from now -- or an
import of one -- would re-couple the two silently. So this now asserts both halves: the v1
modules do not exist, and nothing in v2 names them. Walking the AST is cheap and it fails on
the import itself rather than on whatever breaks later.
"""
from __future__ import annotations

import ast
from pathlib import Path

V2 = Path(__file__).resolve().parents[1] / 'src' / 'roto' / 'v2'
ROTO = V2.parent

V1_ONLY = {
    'model', 'shots',
    'dataset.build', 'dataset.manifest', 'dataset.splits',
}
"""v1's modules, deleted. Neither the files nor an import of them may come back.

``dataset.crop``, ``dataset.arrays`` and ``dataset.layers`` are deliberately absent: they are
version-independent mechanics that v2 reuses as-is, and they survived the deletion. So did
``roto.smoothing``, which was ``roto.model.smoothing`` until v1 was removed -- it is a track
smoother with no v1 in it, and ``dataset.crop`` needs it.
"""


def test_v1_modules_are_gone():
    """The deletion, asserted. A re-added ``roto/model/`` would pass every other test here."""
    back = [f'roto.{m}' for m in sorted(V1_ONLY)
            if ROTO.joinpath(*m.split('.')).with_suffix('.py').exists()
            or (ROTO.joinpath(*m.split('.')) / '__init__.py').exists()]
    assert not back, 'v1 modules are back:\n  ' + '\n  '.join(back)


def _imported_roto_modules(path: Path) -> set[str]:
    """Every ``roto.*`` module one file imports, relative imports resolved."""
    tree = ast.parse(path.read_text(), str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and node.module.startswith('roto.'):
                    out.add(node.module[len('roto.'):])
            else:
                # level 1 is roto.v2, level 2 is roto
                base = '' if node.level >= 2 else 'v2.'
                out.add(f'{base}{node.module}' if node.module else base.rstrip('.'))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith('roto.'):
                    out.add(alias.name[len('roto.'):])
    return out


def test_v2_imports_no_v1_module():
    offences = []
    for path in sorted(V2.rglob('*.py')):
        for mod in _imported_roto_modules(path):
            for banned in V1_ONLY:
                if mod == banned or mod.startswith(banned + '.'):
                    offences.append(f'{path.name} imports roto.{mod}')
    assert not offences, 'v2 imports v1:\n  ' + '\n  '.join(offences)


def test_v2_imports_resolve():
    """Every ``roto.*`` module v2 names actually exists -- catches a typo'd relative import."""
    missing = []
    for path in sorted(V2.rglob('*.py')):
        for mod in _imported_roto_modules(path):
            parts = mod.split('.')
            if not ((ROTO.joinpath(*parts).with_suffix('.py')).exists()
                    or (ROTO.joinpath(*parts) / '__init__.py').exists()):
                missing.append(f'{path.name} -> roto.{mod}')
    assert not missing, 'unresolvable imports:\n  ' + '\n  '.join(missing)


def test_v2_package_is_importable_standalone():
    """Importing v2 must not drag in torch or any v1 module as a side effect."""
    import subprocess
    import sys
    src = str(V2.parents[1])
    code = ('import sys; import roto.v2.dataset, roto.v2.ledger, roto.v2.cli; '
            "bad = [m for m in sys.modules if m.startswith(('roto.model', 'roto.keys', "
            "'roto.shots')) or m == 'torch']; "
            'print(",".join(sorted(bad)))')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         env={'PYTHONPATH': src, 'PATH': '/usr/bin:/bin'})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '', f'v2 pulled in: {out.stdout.strip()}'
