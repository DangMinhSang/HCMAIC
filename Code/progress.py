"""Consistent, configurable tqdm progress for console and Kaggle jobs."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sized
from typing import TypeVar


T = TypeVar("T")

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # local logic tests can still run before dependencies install
    _tqdm = None


def _progress_factory():
    global _tqdm
    if _tqdm is None:
        try:
            from tqdm.auto import tqdm

            _tqdm = tqdm
        except ImportError:
            return None
    return _tqdm


def _enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def progress_enabled() -> bool:
    """Whether model libraries should expose their own progress bars."""
    return _enabled("AIC_PROGRESS") and _progress_factory() is not None


def track(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "item",
    leave: bool = False,
    force: bool = False,
    nested: bool = False,
) -> Iterable[T]:
    """Wrap an iterable with tqdm while suppressing noisy micro-bars by default.

    ``AIC_PROGRESS=0`` disables every bar. ``AIC_PROGRESS_ALL=1`` displays even
    tiny inner loops; otherwise loops shorter than ``AIC_PROGRESS_MIN_ITEMS``
    (default 20) still pass through this helper but do not render a bar.
    """
    if not _enabled("AIC_PROGRESS"):
        return iterable
    show_all = _enabled("AIC_PROGRESS_ALL", "0")
    # This path is called inside OCR and temporal-alignment loops many times.
    # Exit before importing tqdm or parsing thresholds when nested bars are not
    # explicitly requested, keeping observability effectively free by default.
    if nested and not force and not show_all:
        return iterable
    progress_factory = _progress_factory()
    if progress_factory is None:
        return iterable
    if total is None and isinstance(iterable, Sized):
        total = len(iterable)
    try:
        minimum = max(1, int(os.environ.get("AIC_PROGRESS_MIN_ITEMS", "20")))
    except ValueError:
        minimum = 20
    disable = not force and not show_all and total is not None and total < minimum
    if disable:
        return iterable
    return progress_factory(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        mininterval=0.25,
    )


def tracked_range(
    *arguments: int,
    desc: str,
    unit: str = "item",
    leave: bool = False,
    force: bool = False,
    nested: bool = False,
) -> Iterable[int]:
    values = range(*arguments)
    return track(
        values,
        desc=desc,
        total=len(values),
        unit=unit,
        leave=leave,
        force=force,
        nested=nested,
    )
