"""Pure ranking helpers shared by the AIC dashboard pipelines."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


CandidateKey = tuple[str, int]


def candidate_key(item: Any) -> CandidateKey:
    return str(item.video_id), int(item.keyframe_number)


def normalized_scores(values: Sequence[float], *, flat_value: float = 0.5) -> list[float]:
    if not values:
        return []
    minimum, maximum = min(values), max(values)
    spread = maximum - minimum
    if spread < 1e-9:
        return [flat_value] * len(values)
    return [(value - minimum) / spread for value in values]


def select_multisource_candidates(
    visual_results: Sequence[Any],
    ocr_hits: Sequence[Any],
    combined: Mapping[CandidateKey, Any],
    budget: int,
) -> list[Any]:
    """Guarantee Qwen coverage from vision, OCR, metadata, then fused score."""
    budget = max(1, budget)
    selected: list[Any] = []
    seen: set[CandidateKey] = set()

    def add(item: Any | None) -> None:
        if item is None or len(selected) >= budget:
            return
        key = candidate_key(item)
        if key not in seen:
            seen.add(key)
            selected.append(item)

    visual_quota = max(1, round(budget * 0.55))
    ocr_quota = max(1, round(budget * 0.30))
    metadata_quota = max(1, budget // 10)
    for item in visual_results[:visual_quota]:
        add(combined.get(candidate_key(item)))
    for hit in ocr_hits[:ocr_quota]:
        add(combined.get((str(hit.video_id), int(hit.keyframe_number))))
    metadata_ranked = sorted(
        combined.values(),
        key=lambda item: float(getattr(item, "metadata_score", 0.0)),
        reverse=True,
    )
    for item in metadata_ranked[:metadata_quota]:
        add(item)
    for item in sorted(combined.values(), key=lambda value: float(value.score), reverse=True):
        add(item)
        if len(selected) >= budget:
            break
    return selected


def select_diverse_results(
    results: Sequence[Any],
    *,
    limit: int,
    min_frame_gap: int,
    max_per_video: int | None,
) -> list[Any]:
    """Apply submission diversity after all retrieval/reranking stages."""
    selected: list[Any] = []
    frames_by_video: dict[str, list[int]] = {}
    for result in sorted(results, key=lambda item: float(item.score), reverse=True):
        nearby = frames_by_video.setdefault(str(result.video_id), [])
        if max_per_video is not None and len(nearby) >= max_per_video:
            continue
        if any(abs(int(result.frame_id) - frame_id) <= min_frame_gap for frame_id in nearby):
            continue
        nearby.append(int(result.frame_id))
        result.rank = len(selected) + 1
        selected.append(result)
        if len(selected) >= limit:
            break
    return selected
