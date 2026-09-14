"""``roto.v2`` must not import v1, so that v1 can be deleted without touching v2.

The boundary is documented in ``roto/v2/__init__.py``. This test is what makes it true: a
docstring saying "do not import v1" is a wish, and an import added in a hurry six weeks from now
would silently re-couple the two. Walking the AST is cheap and it fails on the import itself
rather than on whatever breaks later.
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
"""Modules v2 replaces or does not need. Deleting these must not break ``roto.v2``.

``dataset.crop``, ``dataset.arrays`` and ``dataset.layers`` are deliberately absent: they are
version-independent mechanics that v2 reuses as-is.
"""


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
