"""Separate scene text from broadcast graphics in AIC keyframes."""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence


OCR_INDEX_SCHEMA_VERSION = 2
DEFAULT_BOTTOM_OVERLAY_START = 0.82

# These phrases are deliberately only a compatibility fallback for old OCR
# indexes that contain no box coordinates. New indexes use geometry instead.
LEGACY_TICKER_PATTERNS = (
    r"\bnguy\s+c[oa]\b",
    r"\bl[uo]?\s+quet\b",
    r"\bv(?:u)?ng\s+nui\b",
    r"\bmien\s+bac\b",
    r"\bmua\s+lon\s+dai\s+nga(?:y)?\b",
    r"\btrung\s+du\b",
    r"\bbac\s+bo\b",
    r"\btrung\s+bo\b",
    r"\bnam\s+bo\b",
    r"\btheo\s+du\s+bao\b",
    r"\bkhu\s+vuc\b",
    r"\btren\s+dia\s+ban\b",
    r"\bthoi\s+tiet\b",
    r"\btin\s+moi\s+nhat\b",
)


def _enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def normalize_overlay_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.lower())
    plain = "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )
    return " ".join(plain.replace("đ", "d").split())


def bottom_overlay_start() -> float:
    """Return the normalized y-coordinate where broadcast bars are ignored."""
    try:
        value = float(os.environ.get("AIC_OCR_BOTTOM_OVERLAY_START", DEFAULT_BOTTOM_OVERLAY_START))
    except ValueError:
        value = DEFAULT_BOTTOM_OVERLAY_START
    return max(0.65, min(value, 0.95))


def _polygon_bounds(polygon: Any) -> tuple[float, float, float, float] | None:
    try:
        points = [(float(point[0]), float(point[1])) for point in polygon]
    except (TypeError, ValueError, IndexError):
        return None
    if len(points) < 2:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def is_broadcast_overlay_box(
    polygon: Any,
    text: str,
    image_width: int,
    image_height: int,
    *,
    start_ratio: float | None = None,
) -> bool:
    """Classify ticker/subtitle boxes by normalized screen position.

    A box must be a normal text line near the bottom edge. Large regions are
    retained because they are more likely to be physical signs/documents than
    a TV lower-third. Small corner station logos and clocks are also ignored.
    """
    if image_width <= 0 or image_height <= 0:
        return False
    bounds = _polygon_bounds(polygon)
    if bounds is None:
        return False
    left, top, right, bottom = bounds
    center_x = (left + right) / (2.0 * image_width)
    center_y = (top + bottom) / (2.0 * image_height)
    relative_width = max(0.0, right - left) / image_width
    relative_height = max(0.0, bottom - top) / image_height
    useful_characters = len(re.sub(r"\W+", "", str(text), flags=re.UNICODE))
    overlay_start = bottom_overlay_start() if start_ratio is None else start_ratio

    bottom_line = (
        center_y >= overlay_start
        and relative_height <= 0.14
        and (relative_width >= 0.10 or useful_characters >= 6)
    )
    corner_graphic = (
        center_y <= 0.20
        and (center_x <= 0.22 or center_x >= 0.78)
        and relative_height <= 0.12
        and relative_width <= 0.32
    )
    return bottom_line or corner_graphic


def legacy_text_quality(text: str, schema_version: int = 1) -> float:
    """Estimate trust for a coordinate-free legacy OCR record.

    Version-2 text has already passed the geometric filter. For version 1,
    several co-occurring weather/news phrases identify the exact class of
    scrolling ticker that caused unrelated broadcast frames to rank first.
    """
    if schema_version >= OCR_INDEX_SCHEMA_VERSION:
        return 1.0
    normalized = normalize_overlay_text(text)
    compact = re.sub(r"\W+", "", normalized)
    if (
        "canhbao" in compact
        and "satlo" in compact
        and re.search(r"nguyi?hiem", compact)
    ):
        # Keep a physical warning sign even when a legacy record also contains
        # a ticker; Qwen will receive only the masked image for all v1 text.
        return 1.0
    matches = sum(bool(re.search(pattern, normalized)) for pattern in LEGACY_TICKER_PATTERNS)
    if matches >= 3:
        return 0.05
    if matches >= 2:
        return 0.12
    return 1.0


def prepare_reranker_image(image_path: str | Path):
    """Load one candidate and strongly blur regions Qwen must not read.

    Sentence Transformers accepts PIL images directly. The masking therefore
    stays in memory for the small rerank pool and never copies the AIC corpus.
    """
    path = Path(image_path)
    if not _enabled("AIC_RERANKER_MASK_OVERLAYS"):
        return str(path)
    from PIL import Image, ImageEnhance, ImageFilter  # type: ignore

    with Image.open(path) as source:
        image = source.convert("RGB").copy()
    width, height = image.size
    if width <= 0 or height <= 0:
        return image
    start = max(0, min(height, int(round(height * bottom_overlay_start()))))
    if start < height:
        region = image.crop((0, start, width, height))
        radius = max(8.0, height * 0.025)
        region = region.filter(ImageFilter.GaussianBlur(radius=radius))
        region = ImageEnhance.Contrast(region).enhance(0.25)
        image.paste(region, (0, start))
    return image


def join_unique_lines(lines: Sequence[str]) -> str:
    return " ".join(dict.fromkeys(line.strip() for line in lines if line.strip()))
