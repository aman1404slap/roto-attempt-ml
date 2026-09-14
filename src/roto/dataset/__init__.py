"""Clean-alpha dataset: top-level layers -> (rendered alpha, spline program) examples.

**Re-exports are lazy (PEP 562), and that is load bearing rather than a style choice.** This
package holds two different kinds of module:

* version-independent mechanics -- ``crop``, ``arrays``, ``layers`` -- which ``roto.v2`` reuses
* v1's own ``build``, ``manifest`` and ``splits``, which v2 replaces and which must be
  deletable without breaking v2 (see ``roto/v2/__init__.py``)

Importing them all eagerly made ``import roto.dataset.crop`` pull in ``roto.shots`` through
``build``, which silently coupled v2 to a module it does not use and cannot keep. Resolving
each name on first access keeps the public API identical for every existing caller while
leaving the two halves independent. ``tests/test_v2_boundary.py`` is what notices if that
stops being true.
"""
from typing import Any

# ``build`` is the one name that collides with a submodule of the same name. Lazy resolution
# cannot win that fight: once ``roto.dataset.build`` is imported anywhere, the import system
# binds the *module* as an attribute of this package, normal lookup succeeds, and __getattr__
# is never consulted. The eager version only worked by ordering -- it rebound the name after
# the submodule import. So ``build`` is fetched from ``roto.dataset.build`` by its two callers
# (``roto.cli``, ``tests/test_dataset.py``) and is listed here for documentation only.
_EXPORTS = {
    'BuiltElement': '.build', 'build': '.build',
    'load_alpha': '.build', 'load_meta': '.build',
    'CropConfig': '.crop', 'CropPlan': '.crop', 'crop_plan': '.crop',
    'RotoLayer': '.manifest', 'LayerStats': '.manifest',
    'discover': '.manifest', 'resolve': '.manifest',
    'DEFAULT_HOLDOUT_EVERY': '.splits', 'SPLIT_VERSION': '.splits',
    'build_splits': '.splits', 'frame_split': '.splits', 'load_splits': '.splits',
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    from importlib import import_module
    return getattr(import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
