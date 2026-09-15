"""Clean-alpha dataset mechanics: crops, arrays and layer selection.

Version-independent machinery that predates v2 and is reused by it as-is. v1's own ``build``,
``manifest`` and ``splits`` lived here too and were deleted when v1 was; ``roto.v2`` replaces
each of them (see ``roto/v2/__init__.py``).

**Re-exports stay lazy (PEP 562).** They were lazy originally because eager ones made
``import roto.dataset.crop`` pull in ``roto.shots`` through ``build``. That coupling is gone
with those modules, but resolving on first access still keeps this package free of import-time
cost for a caller that wants one name from one submodule.
"""
from typing import Any

_EXPORTS = {
    'CropConfig': '.crop', 'CropPlan': '.crop', 'crop_plan': '.crop',
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
